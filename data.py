from torchvision.transforms import Compose, ToTensor, RandomCrop
import argparse
from dataset import DatasetFromFolderEval, DatasetFromFolder, Train_data

def transform1(crop_size=(256,256)):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def transform2():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, crop_size=(256,256)):
    return DatasetFromFolderEval(data_dir, transform=transform1(crop_size))


def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())


