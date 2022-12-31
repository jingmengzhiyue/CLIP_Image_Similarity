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