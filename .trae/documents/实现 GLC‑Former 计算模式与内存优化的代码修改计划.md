## 目标
- 在不改变论文方法（组件与损失、输出路径）的前提下，加入全局注意力的计算模式与内存上限，以解决 OOM，同时保留“严格模式”复现。

## 具体修改
- main_Net.py
  - 新增 CLI 参数：`--glc_global_mode {exact,center,pooled}`，默认 `center`；`--global_tokens_limit` 默认 `8192`（/home/code/unsupervised-light-enhance-ICLR2025-main/main_Net.py:26 附近）。
  - 构建模型后传参：`model.glc_former.global_mode = opt.glc_global_mode`、`model.glc_former.global_tokens_limit = opt.global_tokens_limit`（/home/code/unsupervised-light-enhance-ICLR2025-main/main_Net.py:243–247）。
  - 在训练循环，根据 `opt.lambda_C==0` 或 `opt.mode in {cmfsc, den_phy}`，跳过一致性损失与全局路径调用（保持已有损失开关；位置：/home/code/unsupervised-light-enhance-ICLR2025-main/main_Net.py:155–166）。

- net/lformer_Net.py（GLC‑Former）
  - 在 `GLCFormer.__init__` 增加属性：`global_mode`、`global_tokens_limit`，默认 `center` 与 `8192`。
  - 改造 `_partition_windows`（已改为线性切分）：保留当前实现。
  - 新增获取窗口中心函数：`_get_window_centers(windows)` → `[B, num_windows, C]`（对每窗口 `mean`）。
  - 改造 `_global_attention(x)`：
    - `exact`：仅当 `N ≤ global_tokens_limit` 时使用原实现；若超限，自动转 `pooled` 以避免 OOM。
    - `center`：先取窗口中心 `Xc` 做全局注意力，得到 `Y_global_centers`，再广播到各窗口 token；复杂度 `O(num_windows²)`。
    - `pooled`：对序列做平均池化到 `L ≤ global_tokens_limit` 再做注意力，再线性插值/广播回原序列长度。
  - 前向 `forward(modulated_features)`：
    - 使用新 `_global_attention` 与已实现的局部路径、门控融合 `_feature_fusion`；缓存 `y_global/y_local/alpha` 保持一致性损失读取。

- net/lformer_Net.py（net.forward）
  - 根据 `self.mode` 与 `λ_C` 的有效性：
    - 若 `mode in {cmfsc, den_phy}` 或 `λ_C==0`，仅走局部路径，跳过全局注意力路径（不影响 CM‑FSC 与 DEN‑Phy）。
    - 其它模式按既定逻辑（Hybrid/Full）执行。

## 验证与运行
- 严格论文复现：`--glc_global_mode exact --crop_size 128 --lambda_C 0.1`（避免 OOM）。
- 高效稳定训练：`--glc_global_mode center --global_tokens_limit 8192 --crop_size 256 --lambda_C 0.1`。
- 指标与可视化保持：评估写入 TensorBoard；增强图保存到 `results/val_I/`。

## 交付说明
- 仅修改上述文件与函数，接口向后兼容；默认采用 `center` 模式，避免 OOM；为论文复现提供 `exact` 模式。