## 目标
- 为无监督低光增强训练加入分阶段（warmup→增强介入→增强强化）权重调度，使 CM‑FSC 与 GLC‑Former逐步生效，提升亮度与结构指标。

## 修改点
- main_Net.py：新增 CLI 参数用于阶段断点与各损失权重的阶段序列
  - `--phase_epochs`（例如 `20,60`）
  - `--lambda_F_schedule`（如 `0,0.1,0.2`）
  - `--lambda_C_schedule`（如 `0,0.05,0.1`）
  - `--lambda_exp_schedule`（如 `0.2,0.4,0.4`）
  - `--lambda_color_schedule`（如 `0.05,0.1,0.1`）
  - `--lambda_spa_schedule`（如 `0.1,0.2,0.2`）
  - `--tv_weight_schedule`（如 `0.1,0.05,0.05`）
- main_Net.py：在 `train()` 内根据 `epoch` 选择当前阶段权重
  - 解析上述 schedule 为列表
  - 根据 `epoch` 与 `phase_epochs` 映射到阶段索引
  - 使用该索引选择当前 `lambda_F/C/exp/color/spa` 与 `tv_weight`
- 保持已有 `--mode`、`--glc_global_mode`、`--global_tokens_limit` 生效
- 默认 schedule（当用户未提供时）：
  - phase_epochs=`20,60`
  - F=`0,0.1,0.2`；C=`0,0.05,0.1`
  - exp=`0.2,0.4,0.4`；color=`0.05,0.1,0.1`；spa=`0.1,0.2,0.2`
  - tv=`0.1,0.05,0.05`

## 验证
- 日志打印每 100 iter 同步当前阶段与权重（只显示数值）
- 指标曲线预期：SSIM/LPIPS 在阶段切换后继续改善，亮度更一致

## 运行示例
- `CUDA_VISIBLE_DEVICES=1 python main_Net.py --data_train ... --data_val ... --referance_val ... --mode hybrid --glc_global_mode center --global_tokens_limit 8192 --phase_epochs 20,60 --lambda_F_schedule 0,0.1,0.2 --lambda_C_schedule 0,0.05,0.1 --lambda_exp_schedule 0.2,0.4,0.4 --lambda_color_schedule 0.05,0.1,0.1 --lambda_spa_schedule 0.1,0.2,0.2 --tv_weight_schedule 0.1,0.05,0.05 --crop_size 128 --warmup_epochs 20 --scheduler cosine`