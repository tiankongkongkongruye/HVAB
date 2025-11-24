import os
import shutil
import random
from pathlib import Path
import argparse

def split_dataset(
    source_dir: Path, 
    train_dir: Path, 
    test_dir: Path, 
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    将 source_dir 下的所有文件随机分配到 train_dir 和 test_dir。
    
    Args:
        source_dir (Path): 原始文件夹路径（包含所有待划分的文件）。
        train_dir (Path): 划分出的训练集文件夹路径。
        test_dir  (Path): 划分出的测试集文件夹路径。
        test_ratio (float): 测试集比例，默认为 0.1（即 10%）。
        seed (int): 随机种子，用于可复现划分。
    """
    

    # 获取所有文件（不包含子文件夹）
    all_files = [f for f in source_dir.iterdir() if f.is_file()]
    total = len(all_files)
    n_test = int(total * test_ratio)

    # 打乱并划分
    random.seed(seed)
    random.shuffle(all_files)
    test_files = all_files[:n_test]
    train_files = all_files[n_test:]

    # 确保输出目录存在
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 移动文件
    for f in test_files:
        dest = test_dir / f.name
        shutil.move(str(f), str(dest))
    for f in train_files:
        dest = train_dir / f.name
        shutil.move(str(f), str(dest))

    print(f"总文件数: {total}，测试集: {len(test_files)} 个，训练集: {len(train_files)} 个")

def main():
    parser = argparse.ArgumentParser(
        description="将 middle 文件夹中的文件随机划分为 train/test 子文件夹"
    )
    parser.add_argument(
        '--middle_dir', 
        type=Path, 
        default = "data/Defect",
        help="包含所有待划分文件的 middle 文件夹路径"
    )
    parser.add_argument(
        '--test_ratio', 
        type=float, 
        default=0.1, 
        help="测试集比例（0~1 之间），默认 0.1"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help="随机种子，默认 42"
    )
    args = parser.parse_args()

    src = args.middle_dir.resolve()
    train = src / 'train'
    test = src / 'test'

    if not src.is_dir():
        print(f"错误：{src} 不是有效的文件夹。")
        return

    split_dataset(src, train, test, args.test_ratio, args.seed)

if __name__ == '__main__':
    main()
