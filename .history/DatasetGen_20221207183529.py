import os
import glob
import random
import shutil
from PIL import Image
if __name__ == "__main__":
    test_split_ratio = 0.05
    desired_size=128#图片缩放后的统一大小
    raw_path = "/raw"
    output_train_dir = "/public_datasets/dataset_fruit_veg/train"
    output_test_dir = "/dataset_fruit_veg/test"
    dirs = glob.glob(os.path.join(raw_path,'*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(f'Totally {len(dirs)} classes:{dirs}')
    for path in dirs:
        #对每个类别单独处理
        path = path.split('/')[-1]
        os.makedirs(f'{output_train_dir}/{path}', exist_ok=True)
        os.makedirs(f'{output_test_dir}/{path}', exist_ok=True)
        files = glob.glob(os.path.join(raw_path, path,'*jpg'))
        files += glob.glob(os.path.join(raw_path, path,'*.JPG'))
        files += glob.glob(os.path.join(raw_path,path, '*.png'))
        random.shuffle(files)
        boundary = int(len(files)*test_split_ratio)#训练集和测试集的边界