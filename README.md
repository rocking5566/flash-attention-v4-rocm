# FA4-inspired FMHA on gfx1250

獨立 HIP attention forward：**BF16、原生 BSHD、head dimension 128**。
使用 FA4 conditional rescaling（threshold = 8 base-2 units）、native exp2、
WMMA、TDM double buffering、LDS transpose、分段 operand 預取／IGLP 排程，
PV 依 Q 片段分組、矩陣／softmax 階段的 wave 優先序切換，
以及透過 LDS 重排後的非同步輸出寫回。尚未實作完整 FA4 非同步 pipeline、
TMEM 或獨立 correction warpgroup。設計說明見 [DESIGN.md](DESIGN.md)。

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

Auto 先為 B32/H8、Sq=Sk 且 S512／S2048 選 config 157。其餘形狀依 Sq 選擇：
小於 128 用 1，128–512 用 155，513–2048 用 156，大於 2048 用 150。
156 在 155 的排程上，以第一個有效分數設定 softmax 初始基準，保留首塊 overflow 恢復；
157 再將上述兩種固定尺寸的索引計算簡化。兩者均保留呼叫端傳入的 scale。
完整選擇規則見 [DESIGN.md](DESIGN.md#8-auto-如何選配置)。

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

下表使用程式版本 [`ae25593`](https://github.com/rocking5566/flash-attention-v4-rocm/commit/ae25593aff886828af9f975be836d4214979e936) 的 Auto：S512／S2048 使用 config 157，
S1024 使用 156，S4096–32768 使用 150。CK comparator 明確使用 `qr_tdm_sched` 的
D128、M128/N128、4-wave 配置，兩邊都啟用 expert scheduling。

相對 `e45c742`，156／157 的改動集中在短序列的初始 softmax 基準與固定尺寸索引。
Config 150 的 kernel 機器碼已核對與 `e45c742` 一致；長序列數據來自兩邊重新量測。
另以保存的 `e45c742` binary 與 `ae25593` 在 GPU0 做同卡交錯比較，
S512–2048 的六個 dense／causal case 提升 **1.71–4.06%**。
這組舊版比較獨立記錄，未與下表的 GPU2 數據合併。

以下數據在同一張 gfx1250 A0（HIP_VISIBLE_DEVICES=2）量測，BF16 BSHD、B32/H8/D128。
CK revision 為 `2d3b6e98e4ef99cf7dccf54068f333c32a441348`；整組優化實驗使用相同的 CK binary，
下表的 CK 數據仍與 HIP 一起重新量測。
兩種起跑順序各六筆，每筆固定 warmup=5/repeat=20。TFLOPS 由每邊 12 筆觀察的平均時間計算，
未剔除較慢或較快的樣本。兩邊使用相同 Q/K/V tensors，LSE 關閉，計時期間未啟用 profiler
或額外的 amd-smi 查詢。

| S | Dense HIP / CK TFLOPS | 領先 | Causal HIP / CK TFLOPS | 領先 |
|---:|---:|---:|---:|---:|
| 512 | 1685.58 / 1446.28 | 16.55% | 1106.25 / 859.08 | 28.77% |
| 1024 | 2026.14 / 1773.96 | 14.22% | 1534.92 / 1245.32 | 23.26% |
| 2048 | 2058.57 / 1773.30 | 16.09% | 1776.81 / 1433.03 | 23.99% |
| 4096 | 2030.15 / 1790.10 | 13.41% | 1849.10 / 1541.28 | 19.97% |
| 8192 | 2052.60 / 1829.82 | 12.18% | 1945.72 / 1669.56 | 16.54% |
| 16384 | 2078.82 / 1839.83 | 12.99% | 2011.32 / 1740.05 | 15.59% |
| 32768 | 2085.55 / 1845.09 | 13.03% | 2048.41 / 1767.64 | 15.88% |

表中合併平均的 **14 個 workload 全部超過 10%，其中 3 個超過 20%**；
「大部分 case 領先 20% 以上」的目標尚未達成。S4096 causal 為 19.97%，不計入達標數。
兩種起跑順序各自的 14 個 case 也都超過 10%，但仍有測速波動，不能視為每輪的保證。

程式版本 `ae25593` 通過 440 項一般／固定尺寸正確性比較、224 項特化路徑比較，
以及 72 項 runtime scale 比較，合計 **736 項**，另通過 8 項不匹配尺寸的拒絕檢查。
涵蓋首個分數特別高或低、不同列走不同修正路徑、後續 tile 出現更大分數等情況，
未放寬誤差門檻；每個 benchmark 尺寸另檢查抽樣 rows 的 O 與 LSE。
數據、每筆樣本與 build manifest 保存在未納入版控的 `results/opt21/short-clean/`；
manifest 與驗證用 harness 的 source hashes 已核對與 `ae25593` 一致。

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

ATT 顯示初始基準改動省去了首塊 max 歸約，WMMA 與 exp2 執行次數相同。
這份比較擷取 S2048 dense、SE0/CU1/SIMD0 的 16 個 wave；它支持解釋初始基準的改動，
不能把等待成本的差值直接換算成加速比。Config 157 的固定尺寸收益另以 kernel 計時驗證。
長序列 ATT 的剩餘成本主要落在 WMMA 與 softmax，部分屬於必要的計算工作。
完整比較、沿用的 config 150 證據與限制見 [DESIGN.md](DESIGN.md#10-哪些改善有實測支持)。

## 演算法來源

[FA4 論文](https://arxiv.org/abs/2603.05451) 與
[原作者程式](https://github.com/dao-ailab/flash-attention/tree/2ed4a1f0fc0e36a6ca7b861bfbf86ebb82c4c592)。
Conditional rescaling 保留舊 normalization base，直到新 max 超過它 8 個 base-2 units，
才同步修正累積的 O 和分母；最後用 O/l 正規化。
主要 M128/N128 路徑先檢查非負 exponential partial sum：若每個半列的和不超過 2^8，
就不可能有任何單一值要求超過此門檻的 rescale；只有 guard 失敗才重新計算 max。
132／150／155 仍先找首個 tile 的 max 作為 base；156／157 改用第一個有效分數的
base-2 值加 8，並讓首個 tile 也執行 guard，必要時在乘 V 前修正基準與權重。
分母在兩個半列各自以 FP32 累加，最後才合併。這在精確算術下維持同一 attention，
但有限精度的誤差路徑不同。整體 kernel 加速不等同此技巧的單獨貢獻。
第三方程式的歸屬與授權見 `NOTICE`、`licenses/`。
