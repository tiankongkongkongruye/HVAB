## 问题根因
- 即使设置了 `--mode den_phy`，`net.forward` 仍无条件调用 `self.glc_former(modulated_features)`，导致在 GLC‑Former 的 `_global_attention` 中计算 `N×N` 注意力矩阵并 OOM。

## 热修方案（优先实施）
1) 在 `net/lformer_Net.py` 的 `net.forward` 中按模式跳过 GLC‑Former：
- `mode == 'den_phy'`：不调用 `glc_former`，直接输出 Retinex 路径的 `I`（与论文 DEN‑Phy 一致）。
- `mode == 'cmfsc'`：不调用 `glc_former`，使用 CM‑FSC 映射到图像域作为 `I`。
- `mode == 'hybrid'|'full'`：保留调用 `glc_former`。

2) 在 `main_Net.py` 传参：构建模型后设置 `model.mode = opt.mode`，已完成；确保分支生效后再运行。

## 内存优化方案（保持论文一致性）
3) 为 GLC‑Former 增加计算模式与上限（仅计算近似，不改方法）：
- 新增 CLI：`--glc_global_mode {exact,center,pooled}`（默认 `center`）、`--global_tokens_limit`（默认 `8192`）。
- 在 `GLCFormer.__init__` 增加 `self.global_mode` 与 `self.global_tokens_limit`。
- 改造 `_global_attention(x)`：
  - exact：当 `N ≤ global_tokens_limit` 用原始全 token 注意力；超限则自动退到 pooled。
  - center：对每窗口做均值池化得到中心 token，在中心级做全局注意力，再广播回窗口；复杂度 `num_windows²`。
  - pooled：将序列在 token 维度平均池化到不超过上限，再做全局注意力，最后插值/重复回原长度。
- 保留局部路径与门控融合不变，缓存 `y_global/y_local/alpha` 供一致性损失。

## 运行建议
- 仅 DEN‑Phy（稳定验证，无 GLC）：
  - `CUDA_VISIBLE_DEVICES=0 python main_Net.py --mode den_phy --lambda_F 0 --lambda_C 0 --crop_size 64 --warmup_epochs 5 --tv_weight 0.1 --grad_clip 1.0 --scheduler cosine ...`
- 高效稳定 Full 模式（center 模式避免 OOM）：
  - `CUDA_VISIBLE_DEVICES=0 python main_Net.py --mode full --glc_global_mode center --global_tokens_limit 8192 --lambda_C 0.1 --crop_size 256 --window_size 8 --warmup_epochs 5 --tv_weight 0.1 --grad_clip 1.0 --scheduler cosine ...`
- 严格论文复现（exact 模式，小裁剪）：
  - `CUDA_VISIBLE_DEVICES=0 python main_Net.py --mode full --glc_global_mode exact --lambda_C 0.1 --crop_size 128 --window_size 8 ...`

## 交付
- 修改 `net/lformer_Net.py` 的 `forward` 分支以使 `mode den_phy/cmfsc` 跳过 GLC‑Former。
- 在 `GLCFormer` 增加 `global_mode/global_tokens_limit` 与三种 `_global_attention` 路径。
- 在 `main_Net.py` 增加并接线 CLI 参数；更新默认值保证无改动也可运行。