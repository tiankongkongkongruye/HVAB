"""
import os
from PIL import Image
import torch
import pyiqa  # 安装方式：pip install pyiqa
from torchvision import transforms
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建指标
niqe_metric = pyiqa.create_metric('niqe').to(device)
ma_metric = pyiqa.create_metric('ma').to(device)

# 图像预处理（确保图像大小适用于模型）
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # NIQE 和 Ma-score 通常要求大小统一
    transforms.ToTensor(),
])

# 图片文件夹路径（修改为你的路径）
image_folder = 'results/HTIW/I'

# 初始化结果保存
results = []

# 遍历文件夹中的所有图片
for filename in tqdm(sorted(os.listdir(image_folder))):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            niqe_score = niqe_metric(image_tensor).item()
            ma_score = ma_metric(image_tensor).item()
            pi_score = 0.5 * ((10 - ma_score) + niqe_score)

        results.append({
            'filename': filename,
            'NIQE': niqe_score,
            'Ma': ma_score,
            'PI': pi_score
        })

# 输出每张图像的指标
for res in results:
    print(f"{res['filename']}: NIQE={res['NIQE']:.2f}, Ma={res['Ma']:.2f}, PI={res['PI']:.2f}")

# 计算平均值
avg_niqe = sum(r['NIQE'] for r in results) / len(results)
avg_ma = sum(r['Ma'] for r in results) / len(results)
avg_pi = sum(r['PI'] for r in results) / len(results)

print("\n--- 平均指标 ---")
print(f"平均 NIQE: {avg_niqe:.2f}")
print(f"平均 PI: {avg_pi:.2f}")
print(f"平均 Ma-score: {avg_ma:.2f}")

# 设置路径
    image_folder = "results/HTIW/I"  # 替换为你的图片文件夹路径
    niqe_model_path = "/home/remote_user_1/.cache/torch/hub/pyiqa/niqe_modelparameters.mat"  # 替换为你的NIQE模型路径

"""

import os
import cv2
import pyiqa
import torch
from pathlib import Path
from tqdm import tqdm
import shutil

def main():
    # 设置路径
    # /home/code/CLIP-LIT/inference_result
    # /home/code/FSI/results/TOLED/output_images
    # /home/code/SCI/results/HTIW
    # /home/code/LightenDiffusion/results/unpaired
    # /home/code/Zero-DCE/HTIW_enhanced_results
    image_folder = "results/HTIW_CBAM_RAG/I"  # 替换为你的图片文件夹路径
    niqe_model_path = "/home/remote_user_1/.cache/torch/hub/pyiqa/niqe_modelparameters.mat"  # 替换为你的NIQE模型路径
    
    # 检查文件夹是否存在
    image_dir = Path(image_folder)
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"错误: 文件夹 '{image_folder}' 不存在或不是一个目录")
        return
    
    # 准备设备
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:1")
    print(f"使用设备: {device}")
    
    # 准备NIQE模型缓存路径
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "pyiqa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / "niqe_modelparameters.mat"
    
    # 优化模型文件处理逻辑
    model_file_handled = False
    if niqe_model_path:
        niqe_model_path = Path(niqe_model_path)
        if niqe_model_path.exists():
            # 检查源路径和目标路径是否相同
            if niqe_model_path.resolve() != target_path.resolve():
                print(f"复制NIQE模型文件到缓存目录: {target_path}")
                shutil.copy(niqe_model_path, target_path)
                model_file_handled = True
            else:
                print("NIQE模型文件已在正确位置")
                model_file_handled = True
    
    # 初始化评估指标
    try:
        print("初始化NIQE和PI评估器...")
        niqe_metric = pyiqa.create_metric("niqe").to(device)
        pi_metric = pyiqa.create_metric("pi").to(device)
        print("评估器初始化成功")
    except Exception as e:
        print(f"初始化指标失败: {str(e)}")
        if not model_file_handled:
            print("请确保已正确设置NIQE模型文件路径")
        return
    
    # 获取所有图片文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))  # 大写扩展名
    
    if not image_paths:
        print(f"在 {image_folder} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 初始化统计变量
    total_niqe = 0.0
    total_pi = 0.0
    processed_count = 0
    
    # 处理每张图片
    for img_path in tqdm(image_paths, desc="评估图像质量"):
        try:
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法读取图片 {img_path.name}，跳过")
                continue
                
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 转换为PyTorch tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            
            # 计算指标
            with torch.no_grad():
                niqe_score = niqe_metric(img_tensor).item()
                pi_score = pi_metric(img_tensor).item()
            
            # 累加结果
            total_niqe += niqe_score
            total_pi += pi_score
            processed_count += 1
            
        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {str(e)}")
    
    # 计算平均值
    if processed_count > 0:
        avg_niqe = total_niqe / processed_count
        avg_pi = total_pi / processed_count
        
        print("\n" + "=" * 50)
        print(f"评估完成! 成功处理 {processed_count}/{len(image_paths)} 张图片")
        print(f"平均 NIQE: {avg_niqe:.4f} (值越小表示质量越好)")
        print(f"平均 PI: {avg_pi:.4f} (值越小表示质量越好)")
        print("=" * 50)
    else:
        print("没有成功处理任何图片")

if __name__ == "__main__":
    # 安装必要库的提示
    try:
        import pyiqa
    except ImportError:
        print("请先安装必要库: pip install pyiqa opencv-python tqdm torch torchvision")
        exit(1)
    
    main()