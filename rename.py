import os
import re



def remove_low_prefix(folder_path):
    """批量移除文件名中的low前缀"""
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            # 使用正则表达式匹配数字部分[7](@ref)
            match = re.match(r'low(\d+)\.png', filename)
            if match:
                new_name = f"{match.group(1)}.png"
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_name}")


def remove_normal_prefix(folder_path):
    """批量移除文件名中的low前缀"""
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            # 使用正则表达式匹配数字部分[7](@ref)
            match = re.match(r'normal(\d+)\.png', filename)
            if match:
                new_name = f"{match.group(1)}.png"
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_name}")

# 使用示例
remove_low_prefix('data/LOL-v2/Real_captured/Train/Low/')
remove_normal_prefix('data/LOL-v2/Real_captured/Train/Normal')