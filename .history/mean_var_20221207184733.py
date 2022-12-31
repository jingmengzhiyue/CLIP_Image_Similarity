import os
import glob
import random
import shutil
import numpy as np
from PIL import Image
if __name__ == '__main__':
    dataset_train_dir = ""
    train_files = glob.glob(os.path.join(dataset_train_dir, '*','*jpg'))
    print(f'Totally {len(train_files)}files for training')
    result = []
    for file in train_files:
        img = Image.open(file).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = img/255.
        result.append(img)
    print(np.shape(result))#[BS,H,W,C]
    mean = np.mean(result, axis=(0,1,2))
    std = np.std(result, axis=(0,1,2))
    print(mean)
    print(std)