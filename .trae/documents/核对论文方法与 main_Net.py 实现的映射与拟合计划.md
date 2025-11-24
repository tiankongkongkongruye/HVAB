## 结论
- 与论文的三大组件一一对应：
  - DEN‑Phy（可微 CRF + SDN + 物理一致性）→ RetinexModule 与 physical_loss、retinex_recon_loss（net/lformer_Net.py:787-872；utils_Net.py:118-176、179-224）。
  - CM‑FSC（跨模态频率‑语义协同先验）→ LearnableWaveletTransform + CrossModalModulator + freq_loss（net/lformer_Net.py:433-461、465-536；utils_Net.py:227-248）。
  - GLC‑Former（全局‑局部一致性 Transformer）→ GLCFormer 模块与一致性损失（net/lformer_Net.py:538-783；utils_Net.py:250-264）。
- 差异：当前前向的最终增强 `I` 来自 Retinex 路径，而非“逆小波 + CRF”的完整重建；GLC‑Former仅用于特征正则；跨模态调制简化了论文中的 `V_j·e_j` 部分。

## 代码与论文逐点映射
- 物理损失 L_phy（||Î−I||² + μ||∇Î||₁ + ν(||∇Î_s||₁ + ||∇Î_c||₁)）
  - 实现：utils_Net.py:118-176；在训练拼接：/home/code/unsupervised-light-enhance-ICLR2025-main/main_Net.py:140-146、175-189。
- Retinex重建 L_recon（||R̂◦L̂−I||₁ + ||∇L̂||₁ + ||∇R̂||²）
  - 实现：utils_Net.py:179-224；训练使用：main_Net.py:132-134、175。
- CM‑FSC 频率损失（∑||F̂_j − F_j^clip||²）
  - 实现：utils_Net.py:227-248；特征来源：cross_modal_modulator.modulated_features（net/lformer_Net.py:491-515、883-885）；训练开关：main_Net.py:146-152、165-189。
- GLC‑Former 一致性（门控融合 + 全局/局部一致性）
  - 模块：net/lformer_Net.py:538-783；当前仅缓存 `y_global/y_local` 用于一致性损失（net/lformer_Net.py:768-783、885）；损失：utils_Net.py:250-264；训练开关：main_Net.py:155-166、185-189。
- CRF 与噪声模型（DEN‑Phy）
  - RetinexModule：可微 `kappa` 与噪声 `σ_s,σ_c`，重建 `Î = κ[(R + N(σ_s,σ_c)) ∘ L]`（net/lformer_Net.py:809-872）。

## 当前实现与论文差异的影响
- 终端增强路径：论文建议“逆小波包重建 + CRF”，代码目前“直接使用 Retinex 输出 I”，这更稳但可能限制细节恢复与频率先验的直接作用。
- 跨模态调制：代码使用 `MLP(Concat(avg_F_j, e_img)) ⊙ σ(Q_j e_img/√d)`，省略了 `V_j e_j` 的乘项，建议补全以提高对语义嵌入的适配能力。
- 一致性损失：目前以 `MSE(y_global, y_local)` 近似论文门控一致性，建议将 GLC‑Former 的融合 `Y = α⊙Y_global + (1−α)⊙Interpolate({Y_local^w})` 直接用于输出重建支路或至少用于中间重建特征。

## 拟合计划（不改动接口，逐步逼近论文）
1) 输出路径对齐论文
- 将 GLC‑Former 的融合序列通过完善的逆小波重建为图像特征栈，并与 Retinex 重建结果做门控融合得到最终 `I`。
- 保留 `I_retinex` 作为备选，训练初期在 warmup 阶段使用 Retinex 输出，后期逐步混入 GLC 路径输出（线性权重或学习式门）。

2) 完整化跨模态调制
- 补回 `V_j·e_j` 的仿射投影，并保证 `e_j` 从原始 RGB 经 CLIP 视觉编码获得（已改为 `input_rgb`），使 `F̂_j = F_j + γ_j ⊙ σ((Q_j·e_j)/√d) ⊙ (V_j·e_j)` 与论文一致。
- 将 `modulated_features` 显式用于重建支路，让频率先验实质影响 `I`。频率损失仍与 `F_clip` 对齐（可由 CLIP 视觉引导或经验高频目标）。

3) 一致性与门控机制
- 在 GLC‑Former 内部保留 `α = σ(W_α·[Y_global; Interp(Y_local)])` 的门控，并输出融合的 `Y`，减少单纯 MSE 近似导致的欠拟合。

4) 训练稳态与调参
- 继续使用 warmup（main_Net.py 已支持）与 TV/曝光/颜色/空间损失权重作为稳态正则；在 10–20 个 epoch 之后逐步引入 `λ_F, λ_C`。
- 提供 `--scheduler cosine` 已加入；根据验证曲线调整 `lr` 与权重。

5) 评估与导出
- 保留三指标/两指标的布尔开关；保存验证增强图到 `results/val_I/` 便于质检。

## 交付
- 以上改动保持命令行接口不变，训练脚本使用现有参数即可；改动集中在 `net/lformer_Net.py` 的前向与 GLC‑Former/调制器实现，以及 `main_Net.py` 的输出拼接与损失汇总。
- 完成后将提供行级差异与跑通命令，并在 1–2 个验证周期给出曲线趋势与图像样例供你审阅。