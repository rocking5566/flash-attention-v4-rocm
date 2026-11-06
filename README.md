# FA4-inspired FMHA on gfx1250

獨立 HIP attention forward：**BF16、原生 BSHD、head dimension 128**。
使用 FA4 conditional rescaling（threshold = 8 base-2 units）、native exp2、
WMMA、TDM double buffering 和 LDS transpose。尚未實作完整 FA4 非同步 pipeline、
TMEM 或獨立 correction warpgroup。

## 程式結構

```text
src/bshd.hip             dispatch
src/bshd_kernel.hpp      kernel 與 register helpers
include/fav4_bshd.h      公開 ABI
include/gfx1250.h        TDM／LDS helpers
benchmarks/bshd.py       FP64 驗證、同 GPU 的 TFLOPS 比較
benchmarks/ck_bshd.hip   CK qr_tdm 比較 library
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
HIP/clang（已使用 clang 24）及 ROCm PyTorch；benchmark 的時脈快照使用 `amd-smi`。

獨立 HIP library 不依賴 CK：

```bash
bash scripts/remote.sh bash scripts/build.sh
```

要與 CK 比較，將相容的 rocm-libraries checkout 放在遠端，指定 composablekernel 路徑。
已使用的 CK revision 為 `11c520d2147fab619a5a57e6d7c7368c6b41675b`。
若保有本機 `reference/ck-bshd` snapshot，會隨工作目錄同步，且可省略 `CK_ROOT`。

```bash
bash scripts/remote.sh env CK_ROOT=/work/rocm-libraries/projects/composablekernel \
  bash scripts/build.sh --ck
bash scripts/remote.sh python3 benchmarks/bshd.py validate --output results/validate.json
bash scripts/remote.sh python3 benchmarks/bshd.py bench --output results/bench.json
```

驗證包含 200 次 FP64 output 比較，涵蓋 causal、尾端、多 batch/head 和特殊 logits。
預設 benchmark 使用 [PR #12347](https://github.com/ROCm/rocm-libraries/pull/12347) 尺寸：
B32/H8/D128、S512–32768、dense／causal。可用 `--seqs 1024,4096` 縮小範圍。
每筆固定 **warmup=5、repeat=20**，每邊六筆交錯觀察；兩個 library 收到相同 tensor pointers。
以 HIP graph／GPU events 計時，LSE 關閉、重複使用 buffers，只計 kernel。
Dense FLOPs 為 `4*B*H*D*S²`，causal 為 `4*B*H*D*S*(S+1)/2`。

`build/`、`results/`、CK snapshot、論文 PDF 和個人遠端設定不納入版控。
`results/` 可隨時刪除；建置 manifest 另存於 `build/`，benchmark 不依賴舊 results。
刪除 `build/` 後先重新建置。遠端產物必須在下次同步前取回，`remote.sh` 會處理 `results/`。

## 演算法來源

[FA4 論文](https://arxiv.org/abs/2603.05451) 與
[原作者程式](https://github.com/dao-ailab/flash-attention/tree/2ed4a1f0fc0e36a6ca7b861bfbf86ebb82c4c592)。
Conditional rescaling 保留舊 normalization base，直到新 max 超過它 8 個 base-2 units，
才同步修正累積的 O 和分母；最後用 O/l 正規化。這在精確算術下維持同一 attention，
但有限精度的誤差路徑不同。整體 kernel 加速不等同此技巧的單獨貢獻。
第三方程式的歸屬與授權見 `NOTICE`、`licenses/`。
