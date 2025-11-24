import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from net.Iformer_edit_channelattention import net
from utils import pair_downsampler, image_augmentations  # 确保你已经导入
import argparse

def tensor_to_pil(tensor):
    return TF.to_pil_image(torch.clamp(tensor.squeeze(0).cpu(), 0, 1))

def save_tensor(tensor, path):
    img = tensor_to_pil(tensor)
    img.save(path)

def inference_and_save(input_image_path, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 读取图像并预处理
    image = Image.open(input_image_path).convert('RGB')
    input_tensor = TF.to_tensor(image).unsqueeze(0).to(device)  # Shape: (1,3,H,W)

    # 1. 原图保存
    save_tensor(input_tensor, os.path.join(output_dir, "input.jpg"))

    # 2. 获取im1, im2
    im1, im2 = pair_downsampler(input_tensor)
    save_tensor(im1, os.path.join(output_dir, "im1.jpg"))
    save_tensor(im2, os.path.join(output_dir, "im2_raw.jpg"))

    # 3. 图像增强im2
    im2_aug = image_augmentations(im2)
    save_tensor(im2_aug, os.path.join(output_dir, "im2_aug.jpg"))

    # 4. 模型推理
    with torch.no_grad():
        L1, el1, R1, X1, I1 = model(im1)
        L2, el2, R2, X2, _  = model(im2_aug)

    # 5. 保存模型输出（可选：保存float图为uint8）
    save_tensor(L1, os.path.join(output_dir, "L1.jpg"))
    save_tensor(el1, os.path.join(output_dir, "el1.jpg"))
    save_tensor(R1, os.path.join(output_dir, "R1.jpg"))
    save_tensor(I1, os.path.join(output_dir, "I1.jpg"))

    save_tensor(L2, os.path.join(output_dir, "L2.jpg"))
    save_tensor(el2, os.path.join(output_dir, "el2.jpg"))
    save_tensor(R2, os.path.join(output_dir, "R2.jpg"))

    print(f"All images saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="output_results", help="Directory to save images")
    args = parser.parse_args()

    inference_and_save(args.input_image, args.model_path, args.output_dir)
