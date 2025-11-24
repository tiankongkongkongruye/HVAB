## 目标
- 可直接运行 `main_Net.py` 完成训练与验证（含 TensorBoard 与权重保存）。
- 以 `main_Net.py` 为主线，梳理数据流、模型结构、损失与评估，形成项目解析。
- 先做“可跑通”的最小修复，再逐步完善 CLIP-引导与特征一致性相关模块。

## 运行准备
- Python 与依赖：
  - `pip install torch torchvision lpips tensorboard einops thop pillow opencv-python numpy`
  - CLIP：`pip install git+https://github.com/openai/CLIP.git`
- 硬件：建议 GPU，按需设置 `CUDA_VISIBLE_DEVICES`。
- 数据：
  - 训练低光目录：`--data_train` 指向“单层目录”内的图像集合（不含子文件夹）。
  - 验证低光目录：`--data_val` 同上。
  - 验证参考目录：目前硬编码为 `label_dir`，后续改为参数。

## 快速运行步骤
- 示例命令：
  - 训练+验证：`python main_Net.py --data_train /path/train --data_val /path/val --save_folder weights/run1/ --logroot logs/run1/ --lr 1e-5 --batchSize 1 --nEpochs 100`
  - 指定 GPU：`CUDA_VISIBLE_DEVICES=0 python main_Net.py ...`
- 输出：
  - 权重：`opt.save_folder/epoch_{E}.pth`
  - 指标：`psnr/ssim/lpips` 写入 `opt.logroot`。

## 项目结构与数据/训练流程
- 入口与训练：`main_Net.py`
  - 解析参数→加载数据→构建模型→循环训练→周期性验证与保存。
  - 关键引用：模型 `net.lformer_Net.net`，数据 `data.get_training_set/get_eval_set`，损失 `utils.py`。
- 数据加载：`data.py` + `dataset.py`
  - 训练/验证均使用 `DatasetFromFolderEval`（平铺目录中的图片），`RandomCrop(256)` 仅用于训练。
- 模型：`net/lformer_Net.py`
  - 组件：可学习小波（`LearnableWaveletTransform`）、跨模态调制器（`CrossModalModulator`）、GLC-Former（全局/局部一致性）、RetinexModule（生成 `R/L/I`）。
- 损失：`utils.py`
  - `physical_loss`、`retinex_recon_loss`、`freq_loss`、`consistency_loss`。
- 评估：`PSNR/SSIM`（`main_Net.py:182-230`）、`LPIPS(alex)`。

## 已发现的阻断点（需要修复）
- 缺少必要 import：
  - `cv2`、`numpy`、`PIL.Image` 未在 `main_Net.py` 顶部导入（使用位置：`main_Net.py:182-230, 288-297`）。
  - `clip` 在 `main_Net.py:8` 被注释，但在 `main_Net.py:70-75` 调用；会 `NameError`。
- 硬编码路径：
  - 验证参考 `label_dir` 硬编码于 `main_Net.py:254`，与本地数据不符。
- 模型实现与调用不匹配：
  - `GLCFormer.__init__` 的参数名为 `dim/...`，实例化却传入 `in_channels/out_channels`（`net/lformer_Net.py:888`），将触发 `TypeError`。
  - `CrossModalModulator.forward`：把 16 通道特征图传到 `CLIP.encode_image`（`net/lformer_Net.py:496-501`），不符合 CLIP 接口（需要 3 通道图像）；且 `V_layers[j]` 未调用，后续当张量使用（`net/lformer_Net.py:519-533`）。
  - `net.forward` 返回 `I=enhanced_output`（一个列表，GLC-Former输出，`net/lformer_Net.py:915-917`），但训练中把 `I` 当作张量裁剪（`main_Net.py:93-98`）。
  - 训练中访问 `model.net.cross_modal_modulator.modulated_features`（`main_Net.py:137`），但该属性未在模块内保存。
  - 缺少 `import math`（`net/lformer_Net.py` 使用 `math.sqrt`）。
  - 硬编码 `sys.path.append('/data2/lhq/PairLIE_edit')`（`net/lformer_Net.py:5`）。

## 最小可运行修复方案（Phase 1）
- 在 `main_Net.py` 顶部补充：`import cv2`、`import numpy as np`、`from PIL import Image`、`import clip`。
- 将 `label_dir` 改为 CLI 参数：新增 `--label_dir`，默认指向项目内 `dataset/reference/` 或用户自定义路径；在代码使用该参数替代硬编码（`main_Net.py:254`）。
- 暂时关闭频率与一致性损失（或设权重为 0）：避免依赖未稳定的特征接口。
- `net/lformer_Net.py`：
  - 补充 `import math`。
  - 修正 `GLCFormer` 构造：使用 `GLCFormer(dim=16, ...)`。
  - 调整 `net.forward` 返回：确保 `L/el/R/X/I` 全为张量；建议：`L、R、I` 来自 `RetinexModule`，`el=L`，`X=input`。
  - 在 `GLCFormer.forward` 内部缓存 `y_global/y_local` 到对象属性，供一致性损失读取。
  - 若保留跨模态调制器：先移除对 CLIP 的直接调用或把 CLIP 输入改成原始图像（而非 16 通道特征）；并在对象上缓存 `modulated_features`。

## 逐步增强（Phase 2）
- 重新启用 `freq_loss/consistency_loss`：
  - `freq_loss` 使用 `CrossModalModulator` 的 `modulated_features`（对象属性）。
  - `consistency_loss` 使用 `GLCFormer.y_global/y_local`（对象属性）。
- 将 CLIP 文本提示与图像概率引导（`main_Net.py:70-110`）保留为可选项（参数开关）。

## 验证与解析输出
- 训练日志：每 100 iter 打印损失与 LR；周期性输出 `PSNR/SSIM/LPIPS` 并写入 TensorBoard（`SummaryWriter`）。
- 结构解析交付：
  - 数据流图（文字说明）与关键文件行号引用。
  - 运行截图/命令与指标曲线说明（不改动仓库，仅输出说明）。

## 我将实施的具体改动（获得确认后）
- `main_Net.py`：
  - 增加缺失的 import；新增 `--label_dir` 并使用；将 `lambda_F/lambda_C` 参数化，默认 0。
- `net/lformer_Net.py`：
  - 增加 `import math`；修正 `GLCFormer` 构造；使 `forward` 返回 5 个张量；在对象上保存 `modulated_features` 和 `y_global/y_local`。
- 不改动数据集接口与训练循环的整体形态，确保与原始逻辑一致。

## 需要您确认
- 训练/验证数据与参考图的本地路径（用于 `--data_train/--data_val/--label_dir`）。
- 是否优先以 Phase 1（最小可跑通）为主，随后再启用 CLIP 与一致性相关损失（Phase 2）。