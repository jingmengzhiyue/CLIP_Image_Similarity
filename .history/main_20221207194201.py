import os
import argparse
from matplotlib import pyplot as plt
import glob
import sys
import tqdm
import numpy as np
import torch
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision
from PIL import Image
import timm
import clip
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
def get_args_parser():
    parser = argparse.ArgumentParser('image search task',add_help=False)
    # Model parameters
    parser.add_argument('--input_size',default=128,type=int, help='images input size')
    parser.add_argument('--dataset_dir',default='/Users/alpha/Documents/public_datasets/dataset_fruit_veg/train', help='path where to load images')
    parser.add_argument('--test_image_dir',default='/Users/alpha/Documents/public_datasets/dataset_fruit_veg/val_images', help='images to test,split by comma ","')
    parser.add_argument('--save_dir',default='./output_dir', help='path where to save,empty for no saving')
    parser.add_argument('--model_name',default='resnet50', help='model name')
    parser.add_argument('--feature_dict_file',default='corpus_feature_dict.npy', help='filename where to save image representations')
    parser.add_argument('--topk',default=7,type=int, help='k most similar images to be picked')
    parser.add_argument('-mode',default='extract', help='extract or predict,for extracting features or predicing similar images from corpus')
    return parser


if __name__ == '__main__':
    from pprint import pprint
    model_names = timm.list_models(pretrained=True)
    pprint(model_names)
    args = get_args_parser()
    args = args.parse_args()
    model = None
    preprocess = None
    if args.model_name != "clip":
        model = timm.create_model(args.model_name, pretrained=True)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of trainable params (M):%.2f'%(n_parameters / 1.e6))
        model.eval() 
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model,preprocess = clip.load("ViT-B/32", device=device)
    if args.mode == "extract":
        print(f'use pretrained model {args.model_name} to extract features')
        allVectors = extract_features(args, model, test_image_path=args.dataset_dir, preprocess=preprocess)
    else:
        print(f'use pretrained model {args.model_name} to search {args.topk} similar images from corpus')
        test_images = glob.glob(os.path.join(args.test_image_dir, "*.png"))
        test_images += glob.glob(os.path.join(args.test_image_dir,"*.jpg"))
        test_images += glob.glob(os.path.join(args.test_image_dir,"*.jpeg"))