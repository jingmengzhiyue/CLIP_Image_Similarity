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
    