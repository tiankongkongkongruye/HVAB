## 将进行的代码修改
- 在 net/lformer_Net.py：
  1) 增强 GLC 输出头：将 16 通道特征用小解码器映射到 3 通道图像（Conv→ReLU→Conv→ReLU→Conv→Sigmoid）。
  2) 学习式门控融合：添加 gating_net，输入为亮度均值与 GLC 输出的均值，输出像素级 gate；在 hybrid 模式使用 `I = gate ⊙ I_phy + (1−gate) ⊙ J_img`，full 模式保持 `I = κ^{-1}(J_img)`。
  3) CM‑FSC 目标：为每层增加 T_layers（Linear），由 CLIP嵌入生成 `F_j^clip`，广播到特征图；频率损失目标改为该 `F_j^clip`。
- 在 main_Net.py：
  4) 暴露曝光均值参数：新增 `--exp_mean`，将 `L_exp(48,opt.exp_mean)`。

## 验证
- 保持已有 CLI，不改损失接口；运行 hybrid/full 模式后，亮度与结构指标随阶段应继续改善。

## 说明
- 改动不改变论文方法（组件与损失），只是把融合权重从固定均值改为可学习门控，并增强 GLC 输出表达；频率目标更贴近论文公式。