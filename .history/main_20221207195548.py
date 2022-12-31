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

def extract_feature_single(args, model, file):
    img_rgb = Image.open(file).convert('RGB')
    image = img_rgb.resize((args.input_size, args.input_size), Image.ANTIALIAS)
    image = torchvision.transforms.ToTensor()(image)
    trainset_mean=[0.4754741, 0.43703308, 0.32849099]
    trainset_std=[0.37737389,0.36130483,0.34895992]
    image = torchvision.transforms.Normalize(mean=trainset_mean, std=trainset_std)(image).unsqueeze(0)
    with torch.no_grad():
        features = model.forward_features(image)
        vec = model.global_pool(features)
        vec = vec.squeeze().numpy()
    img_rgb.close()
    return vec

def extract_feature_by_CLIP(model, preprocess, file):
    image = preprocess(Image.open(file)).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(image)
        vec = vec.squeeze().numpy()
    return vec
def extract_features(args, model, test_image_path='', preprocess=None):
    allVectors = {}
    for image_file in tqdm.tqdm(glob.glob(os.path.join(test_image_path,'*','*.jpg'))):
        if args.model_name =="clip":
            allVectors[image_file] = extract_feature_by_CLIP(model, preprocess, image_file)
        else:
            allVectors[image_file] = extract_feature_single(args, model, image_file)
    os.makedirs(f"{args.save_dir}/{args.model_name}",exist_ok=True)
    np.save(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}",allVectors)
    return allVectors
def getsimilarityMatrix(vectors_dict):
    v= np.array(list(vectors_dict.values()))#[NUM,H]
    numerator = np.matmul(v,v.T)#[NUM,NUM]
    denominator = np.matmul(np.linalg.norm(v, axis=1,keepdims=True), np.linalg.norm(v,axis=1,keepdims=True).T)#[NUM,NUM]
    sim = numerator / denominator
    keys = list(vectors_dict.keys())
    return sim, keys

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