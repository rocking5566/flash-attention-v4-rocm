# FA4-inspired FMHA on gfx1250

針對 AMD gfx1250 的獨立 HIP attention forward kernel，使用 **BF16、原生 BSHD layout、head dimension 128**。
以 CK `qr_tdm_sched` 為 baseline，在同一張 GPU、相同輸入下比較效能。

實作採用 FlashAttention-4（FA4）的 conditional rescaling，減少 softmax 的重複修正，
並以 WMMA、TDM double buffering、operand prefetch 與 IGLP 排程，讓資料搬運和計算重疊。
獨立 library 不依賴 CK；CK 只用於比較。這是採用 FA4 部分概念的 gfx1250 實作，
尚未實作完整 FA4 async pipeline。演算法、硬體機制與圖解見 [DESIGN.md](DESIGN.md)，不需要先讀論文。

## 支援範圍

| 項目 | 支援內容 |
|---|---|
| Q / K / V / O | 連續 BF16 `[B, S, H, 128]`；B 為 batch size、S 為 sequence length、H 為 head 數 |
| 計算精度 | FP32 累加與 softmax；輸出為 BF16 |
| Attention | Dense 或 top-left causal，Q / K / V 的 head 數相同 |
| Sequence length | 支援不同 Q / K 長度與 tile 尾端 padding |
| Scale / LSE | 正且有限的 runtime scale；可選 FP32 LSE `[B, H, Sq]` |

不支援 GQA、varlen、paged KV、dropout、bias、sliding window 或 backward。

## 效能紀錄

程式版本 [`85ee282`](https://github.com/rocking5566/flash-attention-v4-rocm/commit/85ee282db96ffb1adcf3d085d295f119f2e23b22)
的 Auto 與 CK [`2d3b6e98`](https://github.com/ROCm/rocm-libraries/commit/2d3b6e98e4ef99cf7dccf54068f333c32a441348)
的 `qr_tdm_sched`，在**同一張 gfx1250 A0 GPU0** 交錯量測。
尺寸依 [ROCm PR #12347](https://github.com/ROCm/rocm-libraries/pull/12347)：
BF16 BSHD、**B32 / H8 / D128、Sq = Sk = S**。

| S | Dense HIP / CK TFLOPS | 領先 | Causal HIP / CK TFLOPS | 領先 |
|---:|---:|---:|---:|---:|
| 512 | 1783.54 / 1468.03 | 21.49% | 1118.61 / 867.87 | 28.89% |
| 1024 | 2143.32 / 1780.00 | 20.41% | 1621.44 / 1256.22 | 29.07% |
| 2048 | 2249.54 / 1850.79 | 21.54% | 1876.98 / 1471.81 | 27.53% |
| 4096 | 2193.40 / 1831.79 | 19.74% | 1966.17 / 1572.05 | 25.07% |
| 8192 | 2171.29 / 1881.41 | 15.41% | 2069.67 / 1704.08 | 21.45% |
| 16384 | 2212.69 / 1887.22 | 17.25% | 2139.36 / 1776.56 | 20.42% |
| 32768 | 2221.52 / 1895.33 | 17.21% | 2193.21 / 1805.68 | 21.46% |

**14 個 workload 全部領先 10% 以上，其中 10 個超過 20%。** 所有 causal 均超過 20%；
S4096–32768 dense 為 15.41–19.74%。接近 20% 的 case 仍會波動：
兩種起跑順序分別有 11／14、9／14 超過 20%，表格呈現兩者合併的結果。

每筆固定 **warmup=5、repeat=20**，兩種起跑順序各六筆，依每邊共 12 筆的平均時間計算，
不剔除樣本。兩邊共用 Q/K/V/O tensors，以 HIP graph／GPU events 計時，重複使用 buffers、
關閉 LSE，只計 kernel；計時期間沒有 profiler 或額外的 `amd-smi` 查詢。

<details>
<summary>量測環境、計算方式與驗證</summary>

- 兩邊使用 clang 24 `d6f6cb691863`、`gfx1250-strict`、HIP runtime 7.16.26385，
  環境為 ROCm Python SDK 10.1.0、PyTorch 2.14.0+rocm10.1.0。
- HIP Auto 在這七種尺寸選 config 158；CK 使用 D128、M128/N128、4-wave 配置。
  整組實驗固定 CK source、編譯旗標與 binary，每次仍與 HIP 重新計時。
- 兩邊均啟用 expert scheduling；HIP 另使用 `max-ilp` 與 barrier latency cost 32。
  這是編譯器的排程成本模型，並非插入 32 cycles 等待。完整旗標見 [build.sh](scripts/build.sh)。
- Dense FLOPs 為 `4*B*H*D*S²`，causal 為 `4*B*H*D*S*(S+1)/2`；
  領先百分比為 `(CK 平均時間 / HIP 平均時間 - 1) * 100%`。
- `85ee282` 通過 1568 項數值比較、20 項參數拒絕檢查及原生 TDM 驗證。
  高精度參考僅用於驗證，不計入 kernel 時間。

誤差門檻、ATT 分析與驗證範圍見 [DESIGN.md](DESIGN.md#10-哪些改善有實測支持)。
更換 GPU、ISA target、compiler 或 runtime 後，需在同一環境重新量測兩邊。

</details>

## 建置與測試

本機編輯與操作 git，遠端 GPU container 負責編譯、驗證與測速。
先從 [.remote.env.example](.remote.env.example) 建立 `.remote.env`，填入 host、container、
同步路徑與 GPU 編號。`scripts/remote.sh` 會同步程式、啟動 container、執行命令並取回 `results/`。

本機需要 SSH 與 rsync；遠端需要支援 gfx1250 builtins 的 HIP/clang，
執行 Python 驗證與 benchmark 另需相容的 ROCm PyTorch。

### 建置獨立 HIP library

```bash
bash scripts/remote.sh bash scripts/build.sh
```

產生 `build/libfav4_bshd.so`，並執行原生 TDM 驗證，不需要 CK checkout。
預設 compiler 為 `hipcc`、target 為 `gfx1250`；可透過 `HIPCC`、`FAV4_ARCH` 指定。

<details>
<summary>重現效能表的 gfx1250 A0 工具鏈設定</summary>

效能表使用 `gfx1250-strict`。在該 A0 環境，一般 `gfx1250` 載入路徑會插入額外 TDM 等待，
因此重現時需讓兩邊使用相同 ISA target 與相容的 compiler / runtime。
將以下路徑替換成遠端 container 內的工具鏈與 CK checkout：

```bash
bash scripts/remote.sh env FAV4_ARCH=gfx1250-strict \
  HIPCC=/path/to/clang++ CLANG_OFFLOAD_BUNDLER=/path/to/clang-offload-bundler \
  CK_ROOT=/work/rocm-libraries/projects/composablekernel \
  bash scripts/build.sh --ck
```

`build.sh` 保留 strict ELF 的 ISA，只將外層 HIP bundle selector 設為 runtime 回報的 `gfx1250`，
並在 manifest 記錄兩者。`HIPCC` 應指向原始 compiler 或僅設定路徑的 wrapper，避免重複轉換 bundle。
ROCm Python SDK 也可使用其 `hipcc`；若 Python 找不到 `libamdhip64.so`，
在遠端 container 的執行環境加入 SDK library 路徑：

```bash
export LD_LIBRARY_PATH="$(python3 -m rocm_sdk path --root)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

</details>

### 驗證與 CK 比較

Python 驗證與 benchmark 會載入兩個 library，需先建置 CK comparator。
將相容的 rocm-libraries checkout 放在遠端，並以 `CK_ROOT` 指向 container 內的
`projects/composablekernel`。若要重現效能表，使用上方指定的 CK revision 與工具鏈。
若已依 A0 設定建置兩個 library，可直接從 `validate` 開始。

```bash
bash scripts/remote.sh env CK_ROOT=/work/rocm-libraries/projects/composablekernel \
  bash scripts/build.sh --ck
bash scripts/remote.sh python3 benchmarks/bshd.py validate --output results/validate.json
bash scripts/remote.sh python3 benchmarks/bshd.py bench --no-telemetry \
  --order-phase 0 --output results/phase0.json
bash scripts/remote.sh python3 benchmarks/bshd.py bench --no-telemetry \
  --order-phase 1 --output results/phase1.json
```

Benchmark 預設涵蓋表格的 14 個 workload，使用 Auto；可加 `--seqs 1024,4096` 縮小範圍。
每個 workload 分別合併兩份 JSON 的 `samples_us`，由平均時間計算 TFLOPS 與領先百分比。
`--no-telemetry` 跳過額外的 `amd-smi` 查詢，計時方式不變。

`reference/ck-bshd` 是可選的本機 CK source snapshot，有它時可省略 `CK_ROOT`；
它不納入版控，從 GitHub clone 後需自行提供 CK checkout 才能執行比較。
`results/` 可刪除，不影響建置或執行；`build/` 保存 binary 與 manifest，刪除後需重新建置。
兩者及個人 `.remote.env` 均不納入版控。

## API 與程式導覽

以 [Fav4Args](include/fav4_bshd.h) 提供 device pointers、尺寸與 scale，使用呼叫端的 HIP stream：

```cpp
int status = fav4_bshd_forward(&args, causal, 0, stream); // config 0 = Auto
```

函式不配置記憶體，也不做 host synchronization。Pointer 對齊、stream 與輸入限制見
[公開 header](include/fav4_bshd.h)；Auto 選擇規則見 [DESIGN.md](DESIGN.md#8-auto-如何選配置)。

| 入口 | 內容 |
|---|---|
| [DESIGN.md](DESIGN.md) | Attention 基礎、kernel 資料流、優化原理與 ATT 證據 |
| [src/bshd.hip](src/bshd.hip) | Auto dispatch 與 kernel 啟動 |
| [src/bshd_kernel.hpp](src/bshd_kernel.hpp) | QK、softmax、PV 與資料搬運排程 |
| [include/gfx1250.h](include/gfx1250.h) | TDM、LDS transpose、async store helpers |
| [benchmarks/bshd.py](benchmarks/bshd.py) | 正確性驗證與同 GPU 測速 |
| [benchmarks/ck_bshd.hip](benchmarks/ck_bshd.hip) | CK `qr_tdm_sched` comparator |

## 來源與授權

演算法參考 [FA4 論文](https://arxiv.org/abs/2603.05451) 與
[原作者程式](https://github.com/dao-ailab/flash-attention/tree/2ed4a1f0fc0e36a6ca7b861bfbf86ebb82c4c592)。
部分底層描述與索引映射參考 CK；第三方程式歸屬及授權見 [NOTICE](NOTICE) 與 [licenses/](licenses/)。
