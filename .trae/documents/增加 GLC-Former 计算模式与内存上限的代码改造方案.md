## 目标
- 在不改变论文方法（组件、损失、输出路径）的前提下，新增全局注意力计算模式与内存上限，默认使用高效稳定的 center 模式，保留 exact 模式以严格复现。

## 需要新增的 CLI 参数（main_Net.py）
- 在参数定义处（约 `main_Net.py:24-45`）添加：
  - `parser.add_argument('--glc_global_mode', type=str, choices=['exact','center','pooled'], default='center', help='GLC-Former global attention mode')`
  - `parser.add_argument('--global_tokens_limit', type=int, default=8192, help='Max tokens for global attention; pooled when exceeded')`
- 在模型构建后（约 `main_Net.py:243-247`）接线：
  - `model.glc_former.global_mode = opt.glc_global_mode`
  - `model.glc_former.global_tokens_limit = opt.global_tokens_limit`
  - 已有 `--window_size`：`model.glc_former.window_size = opt.window_size`

## GLC-Former 改造（net/lformer_Net.py）
- 在 `GLCFormer.__init__` 增加属性默认值：
  - `self.global_mode = 'center'`
  - `self.global_tokens_limit = 8192`
- 新增工具函数：
  - `_get_window_centers(windows)`：对 `windows ∈ [B,num_windows,window_tokens,C]` 在 `window_tokens` 维度做均值，得到 `[B,num_windows,C]`
- 改造 `_global_attention(x)` 为模式分支：
  - exact：
    - 若 `N <= self.global_tokens_limit`，使用原始全 token 注意力 `Softmax(QK^T/√C)V ⊙ X`
    - 若超限，自动退到 pooled
  - center：
    - `windows, original_n = _partition_windows(x)`
    - `Xc = _get_window_centers(windows)`，对 `Xc` 做全局注意力得到 `Yc`
    - 将 `Yc` 广播到各窗口 token，拼接成 `Y_global ∈ [B, original_n, C]`
  - pooled：
    - 将 `x` 在 token 维度平均池化到 `L = min(N, self.global_tokens_limit)`
    - 对池化序列做全局注意力得到 `Y_pool`，线性插值或重复回原长度 `N` 得到 `Y_global`
- 保持局部路径与门控融合 `_feature_fusion(y_global,y_local)` 不变；沿用现有 `_partition_windows` 与 `_interpolate_local_features` 的线性切分与裁剪。

## 前向路径开关（net.forward 或 GLCFormer.forward）
- 当 `self.mode in {'cmfsc','den_phy'}` 或训练阶段 `lambda_C==0`（通过标志传入或读取）时，跳过 `_global_attention`，仅使用局部路径与逆重建，以避免无用计算与显存占用。
- 其它模式按既定逻辑执行（Hybrid/Full），输出与损失不变。

## 验证与运行
- 高效稳定（默认）：
  - `CUDA_VISIBLE_DEVICES=0 python main_Net.py --data_train ... --data_val ... --referance_val ... --mode full --glc_global_mode center --global_tokens_limit 8192 --lambda_F 0.2 --lambda_C 0.1 --window_size 8 --warmup_epochs 5 --tv_weight 0.1 --grad_clip 1.0 --scheduler cosine --crop_size 256`
- 严格论文复现（小裁剪避免 OOM）：
  - `... --glc_global_mode exact --global_tokens_limit 8192 --lambda_C 0.1 --crop_size 128`
- 评估与保存：保持现有三指标/两指标评估与 `results/val_I/` 可视化。

## 影响与一致性说明
- 改动仅限注意力的计算近似与上限控制，不改变论文组件、损失或输出路径；`exact` 模式确保方法学一致性，`center/pooled` 作为工程优化以避免显存爆炸。