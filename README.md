# FA4-inspired FMHA on gfx1250

獨立 HIP attention forward：**BF16、原生 BSHD、head dimension 128**。
使用 FA4 conditional rescaling（threshold = 8 base-2 units）、native exp2、
WMMA、TDM double buffering、LDS transpose、分段 operand 預取／IGLP 排程，
以及透過 LDS 重排後的非同步輸出寫回。尚未實作完整 FA4 非同步 pipeline、
TMEM 或獨立 correction warpgroup。

## 程式結構

```text
src/bshd.hip             dispatch
src/bshd_kernel.hpp      kernel 與 register helpers
include/fav4_bshd.h      公開 ABI
include/gfx1250.h        TDM／LDS helpers
benchmarks/bshd.py       BF16 正確性驗證、同 GPU 的 TFLOPS 比較
benchmarks/ck_bshd.hip   CK qr_tdm_sched 比較 library
tests/bshd_tdm.hip       TDM stride／padding／zero-fill 驗證
scripts/build.sh         建置；--ck 加入 CK 比較 library
scripts/remote.sh        依 .remote.env 同步、執行、取回產物
```

Q/K/V/O 為連續 `[B,S,H,128]`，可選 LSE 為 FP32 `[B,H,Sq]`。支援 equal heads、
不同 Q/K 長度、尾端 padding、dense 或 top-left causal；scale 必須為正且有限。
不支援 GQA、varlen、paged KV、dropout、bias、sliding window 或 backward。
完整 pointer／stream 契約與配置 ID 見 `include/fav4_bshd.h`。

```cpp
int status = fav4_bshd_forward(&args, causal, 0, stream); // config 0 = Auto
```

## 建置與驗證

依 `AGENTS.md`，在本機修改，在遠端編譯／測試。複製 `.remote.env.example` 為
`.remote.env` 並填入 GPU host、container 和裝置編號。需要支援 gfx1250 builtins 的
HIP/clang（已使用 clang 24，建置預設啟用 expert scheduling）及 ROCm PyTorch；benchmark 的時脈快照使用 `amd-smi`。

獨立 HIP library 不依賴 CK：

```bash
bash scripts/remote.sh bash scripts/build.sh
```

要與 CK 比較，將相容的 rocm-libraries checkout 放在遠端，指定 composablekernel 路徑。
效能紀錄使用的 CK snapshot 取自 upstream develop 的 `2d3b6e98e4ef99cf7dccf54068f333c32a441348`。
它只包含已提交的版本，不包含 rocm-libraries 工作目錄中的未提交修改。
若保有本機 `reference/ck-bshd` snapshot，會隨工作目錄同步，且可省略 `CK_ROOT`。

```bash
bash scripts/remote.sh env CK_ROOT=/work/rocm-libraries/projects/composablekernel \
  bash scripts/build.sh --ck
bash scripts/remote.sh python3 benchmarks/bshd.py validate --output results/validate.json
bash scripts/remote.sh python3 benchmarks/bshd.py bench --output results/bench.json
```

在下述實測的 A0 主機上，兩邊都以 `FAV4_ARCH=gfx1250-strict` 建置，並使用相容的
clang 24 與 HIP 7.16。可將以下環境變數傳給上述建置命令：

```bash
FAV4_ARCH=gfx1250-strict HIPCC=/path/to/clang++ \
  CLANG_OFFLOAD_BUNDLER=/path/to/clang-offload-bundler \
  bash scripts/build.sh --ck
```

此命令在遠端 container 執行，路徑需指向該機的工具鏈。`build.sh` 會保留 strict
ELF 的 ISA，只將外層 HIP bundle selector 設為 runtime 回報的 `gfx1250`，
並在 manifest 記錄兩者。`HIPCC` 應指向原始編譯器或單純設定路徑的 wrapper，
不要另外包一層 strict bundle 轉換。一般 `gfx1250` 建置不經這個包裝流程。

若使用 ROCm Python SDK（實測使用 10.1.0），可直接使用其 `hipcc`。
若 benchmark 的 `ctypes` 找不到 `libamdhip64.so`，在遠端執行 benchmark 前設定：

```bash
export LD_LIBRARY_PATH="$(python3 -m rocm_sdk path --root)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

BF16 正確性驗證涵蓋 causal、尾端、多 batch/head 和特殊 logits。
高精度參考僅供檢查誤差，不計入 kernel 時間；kernel 輸入／輸出均為 BF16，累加與 softmax 為 FP32。
預設 benchmark 使用 [PR #12347](https://github.com/ROCm/rocm-libraries/pull/12347) 尺寸：
B32/H8/D128、S512–32768、dense／causal。可用 `--seqs 1024,4096` 縮小範圍。
可加 `--no-telemetry` 跳過額外的 amd-smi 時脈／溫度／功耗查詢；計時方式不變。
每筆固定 **warmup=5、repeat=20**，每邊六筆交錯觀察；兩個 library 收到相同 tensor pointers。
以 HIP graph／GPU events 計時，LSE 關閉、重複使用 buffers，只計 kernel。
Dense FLOPs 為 `4*B*H*D*S²`，causal 為 `4*B*H*D*S*(S+1)/2`。

`build/`、`results/`、CK snapshot、論文 PDF 和個人遠端設定不納入版控。
`results/` 可隨時刪除；建置 manifest 另存於 `build/`，benchmark 不依賴舊 results。
刪除 `build/` 後先重新建置。遠端產物必須在下次同步前取回，`remote.sh` 會處理 `results/`。

## 效能紀錄

Auto 在 Sq<128 使用 config 1，Sq≤2048 使用 config 132，較長 Q 使用 config 145。
CK comparator 明確使用
`qr_tdm_sched` 的 D128、M128/N128、4-wave 配置，兩邊都啟用 expert scheduling。

Config 132/133 使用條件式 max 計算、延後合併分母、協作式初始 TDM 載入，
以及 LDS 重排／非同步輸出。132 的工作群組順序針對短序列調整；145 在 133 上加入
QK register allocation spacing。此配置對編譯器敏感，必須重新量測，不能視為 bank conflict 已解決。

以下數據對應本 repo 的程式版本 [`58716a2`](https://github.com/rocking5566/flash-attention-v4-rocm/commit/58716a2c1c580309b0444305cf61293d6cb575a8)。兩邊在同一張 gfx1250 A0
（HIP_VISIBLE_DEVICES=2）重新建置與量測，BF16 BSHD、B32/H8/D128。
CK revision 為 `2d3b6e98e4ef99cf7dccf54068f333c32a441348`；其 qr_tdm_sched
pipeline 與 tile 設定相較 CK `11c520d2147fab619a5a57e6d7c7368c6b41675b` 未變，相關 core headers 已更新。
兩種起跑順序各六筆，每筆固定 warmup=5/repeat=20。TFLOPS 由每邊 12 筆觀察的平均時間計算，
未剔除較慢或較快的樣本。兩邊使用相同 Q/K/V tensors，LSE 關閉，計時期間未啟用 profiler
或額外的 amd-smi 查詢。所有數據來自該次同 GPU 比較。

| S | Dense HIP / CK TFLOPS | 領先 | Causal HIP / CK TFLOPS | 領先 |
|---:|---:|---:|---:|---:|
| 512 | 1632.81 / 1465.13 | 11.44% | 1037.94 / 863.79 | 20.16% |
| 1024 | 1955.76 / 1770.86 | 10.44% | 1495.08 / 1236.90 | 20.87% |
| 2048 | 1993.57 / 1783.72 | 11.76% | 1713.99 / 1441.07 | 18.94% |
| 4096 | 1996.66 / 1795.26 | 11.22% | 1815.41 / 1545.73 | 17.45% |
| 8192 | 2026.16 / 1827.64 | 10.86% | 1913.82 / 1671.33 | 14.51% |
| 16384 | 2050.29 / 1839.70 | 11.45% | 1975.01 / 1736.70 | 13.72% |
| 32768 | 2059.20 / 1845.62 | 11.57% | 2009.50 / 1768.08 | 13.65% |

表中合併平均的 14 個 workload 全部超過 10%；反向起跑順序的 S1024／S4096 dense
分別領先 8.96%／9.50%，不能解讀成每輪都保證超過 10%。
8K–32K 的 dense／causal 在兩種順序都超過 10%。
上述 CK 版本與 Auto 通過 112 項 BF16 正確性比較，包含 partial-sum guard 邊界與 overflow／mixed-row 案例；
每個 benchmark 尺寸另檢查抽樣 rows 的 O 與 LSE，BF16 TDM stride／padding／zero-fill 也通過。
數據、每筆樣本、source snapshot 與 build manifest 保存在未納入版控的 `results/`；
manifest 的 source hashes 已核對與程式版本 `58716a2` 一致。

兩邊使用 clang 24 `d6f6cb691863`、`gfx1250-strict` 與 HIP runtime 7.16.26385；
實測環境為 ROCm Python SDK 10.1.0、PyTorch 2.14.0+rocm10.1.0。
A0 的一般 `gfx1250` 載入路徑會插入額外 TDM 等待，因此不能用不同 ISA target 或 runtime 的數字交叉比較。

可用下列命令重測兩種順序，執行環境需使用與建置相容的 HIP/PyTorch：

```bash
python3 benchmarks/bshd.py bench --no-telemetry --order-phase 0 --output results/phase0.json
python3 benchmarks/bshd.py bench --no-telemetry --order-phase 1 --output results/phase1.json
```

每個尺寸分別合併兩個檔案的 `samples_us`，以 CK 平均時間除以 HIP 平均時間得到表中的 speedup。
`--min-speedup 1.10` 檢查的是該次執行的門檻；單一順序可能因主機狀態而未達標。
效能與主機狀態有關，其他機器必須重新量測兩邊。

程式版本 `58716a2` 的 8K dense ATT 由 rocprofv3 擷取並以 rcv-cli 分析，保留 raw trace、source 與 binary hashes。
上述 CK 版本的比較僅重跑正確性與效能；ATT 沿用該程式版本的獨立擷取紀錄。
64 個取樣 wave 沒有 `s_wait_idle`，非同步輸出 store 數量符合預期；剩餘 exposed attribution
主要是 WMMA／softmax，不能將此歸因直接換算成 GPU 利用率或加速比。更多 TDM issuer 與較小
IGLP 群組未帶來實測改善，已移除。分析產物在未納入版控的 `results/att-final/`。

## 演算法來源

[FA4 論文](https://arxiv.org/abs/2603.05451) 與
[原作者程式](https://github.com/dao-ailab/flash-attention/tree/2ed4a1f0fc0e36a6ca7b861bfbf86ebb82c4c592)。
Conditional rescaling 保留舊 normalization base，直到新 max 超過它 8 個 base-2 units，
才同步修正累積的 O 和分母；最後用 O/l 正規化。
首個 tile 計算 max，後續先檢查非負 exponential partial sum：若每個半列的和不超過 2^8，
就不可能有任何單一值要求超過此門檻的 rescale；只有 guard 失敗才重新計算 max。
分母在兩個半列各自以 FP32 累加，最後才合併。這在精確算術下維持同一 attention，
但有限精度的誤差路徑不同。整體 kernel 加速不等同此技巧的單獨貢獻。
第三方程式的歸屬與授權見 `NOTICE`、`licenses/`。
