#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:13:34 2018

@author: cfchen
"""

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os

from helpers import makedir
import model
import find_nearest
import train_and_test as tnt
#import save
#from log import create_logger
#from preprocess import mean, std, preprocess_input_function
from preprocess import mean, std, preprocess_input_function, img_size
from node import Node
import argparse

img_size = 224 #model.img_size

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
parser.add_argument('--data_path', default="../datasets/imagenet/", type=str)
parser.add_argument('--model_name', default="last", type=str)
parser.add_argument('--resume_path', default=None, type=str) # to the dir, not .pth
parser.add_argument('--batch_size', default=15, type=int)
parser.add_argument('--n_protos_per_class', default=8, type=int)
parser.add_argument('--proto_dim', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]


# load the data
data_path = args.data_path
train_dir = data_path + 'train/'
valid_dir = data_path + 'valid/'
test_dir = data_path + 'test/'

batch_size = args.batch_size

# dataset setup

transform_push = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

transform_test_norm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transform_push)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False)

# valid set
valid_dataset = datasets.ImageFolder(
    valid_dir,
    transform_test_norm)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False)

# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transform_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)




# set up directories
root_dir_for_saving_train_images = os.path.join(args.resume_path, 'nearest_train')
root_dir_for_saving_test_images = os.path.join(args.resume_path, 'nearest_test')
makedir(root_dir_for_saving_train_images)
makedir(root_dir_for_saving_test_images)

root = Node("root")
root.add_children(['animal','vehicle','everyday_object','weapon','scuba_diver'])
root.add_children_to('animal',['non_primate','primate'])
root.add_children_to('non_primate',['African_elephant','giant_panda','lion'])
root.add_children_to('primate',['capuchin','gibbon','orangutan'])
root.add_children_to('vehicle',['ambulance','pickup','sports_car'])
root.add_children_to('everyday_object',['laptop','sandal','wine_bottle'])
root.add_children_to('weapon',['assault_rifle','rifle'])
root.assign_all_descendents()

flat_root = Node("root")
flat_root.add_children(['scuba_diver','African_elephant','giant_panda','lion','capuchin','gibbon','orangutan','ambulance','pickup','sports_car','laptop','sandal','wine_bottle','assault_rifle','rifle'])
flat_root.assign_all_descendents()

img_dirs_train = [os.path.join(root_dir_for_saving_train_images,name) for name in root.classes_with_children()]
img_dirs_test = [os.path.join(root_dir_for_saving_test_images,name) for name in root.classes_with_children()]
for img_dir in img_dirs_train:
    makedir(img_dir)
for img_dir in img_dirs_test:
    makedir(img_dir)



# construct the model
resume_path = os.path.join(args.resume_path,"best_model_" + args.model_name + "_opt.pth")
vgg = model.vgg16_proto(flat_root, pretrained=True, num_prototypes_per_class=args.n_protos_per_class, prototype_dimension = args.proto_dim, img_size=img_size, resume_path = resume_path)
vgg = vgg.cuda()
vgg_multi = torch.nn.DataParallel(vgg)
class_specific = True


# label2name
class_names = os.listdir(train_dir)
class_names.sort()
label2name = {i : name for (i,name) in enumerate(class_names)}


# check that this is the right model
# test_acc = tnt.test(model=vgg_multi, dataloader=valid_loader, label2name=label2name, class_specific=class_specific, log=print, class_acc = False)

k = 5

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=vgg_multi, # pytorch network with prototype_vectors
        k=k+1,
        preprocess_input_function=preprocess_input_function, # normalize if needed)
        prototype_layer_stride=1,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        save_image_class_identity=True,
        log=print)

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=vgg_multi, # pytorch network with prototype_vectors
        k=k,
        preprocess_input_function=preprocess_input_function, # normalize if needed)
        prototype_layer_stride=1,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        save_image_class_identity=True,
        log=print)
