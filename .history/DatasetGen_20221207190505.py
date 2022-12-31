import os
import glob
import random
import shutil
from PIL import Image
if __name__ == "__main__":
    test_split_ratio = 0.20
    desired_size=128#图片缩放后的统一大小
    raw_path = "./Dataset/raw"
    output_train_dir = "./Dataset/train/"
    output_test_dir = "./Dataset/test/"
    dirs = glob.glob(os.path.join(raw_path,'*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(f'Totally {len(dirs)} classes:{dirs}')
    for path in dirs:
        #对每个类别单独处理
        path = path.split('\\')[-1]
        os.makedirs(f'{output_train_dir}/{path}', exist_ok=True)
        os.makedirs(f'{output_test_dir}/{path}', exist_ok=True)
        files = glob.glob(os.path.join(raw_path, path,'*.jpg'))
        files += glob.glob(os.path.join(raw_path, path,'*.JPG'))
        files += glob.glob(os.path.join(raw_path,path, '*.png'))
        random.shuffle(files)
        boundary = int(len(files)*test_split_ratio)#训练集和测试集的边界
        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')
            old_size = img.size #old size[0]is in (width,height)format
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            im = img.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new("RGB",(desired_size, desired_size))
            new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
            assert new_im.mode =='RGB'
            if i <= boundary:
                new_im.save(os.path.join(f'{output_test_dir}/{path}', file.split('/')[-1].split('.')[0]+'.jpg'))
            else:
                new_im.save(os.path.join(f'{output_train_dir}/{path}',file.split('/')[-1].split('.')[0]+'.jpg'))
test_files = glob.glob(os.path.join(output_test_dir,'*','*.jpg'))
train_files = glob.glob(os.path.join(output_train_dir, '*','*.jpg'))
print(f'Totally {len(train_files)}files for training')
print(f'Totally {len(test_files)}files for test')