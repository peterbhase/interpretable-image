import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import transforms
import torchvision.datasets as datasets

import argparse

from helpers import makedir, adjust_learning_rate
import model
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, img_size, preprocess_input_function
from node import Node


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
#parser.add_argument('--data_path', default="/usr/xtmp/peterhas/datasets/imagenetIDsmall/", type=str)
parser.add_argument('--data_path', default="../datasets/imagenet/", type=str)
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--resume_path', default=None, type=str) # to the dir, not .pth
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--batch_mult', default=1, type=int)
parser.add_argument('--num_classes', default=15, type=int)
parser.add_argument('--latent_dim', default=512, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
#print('training with %d gpu(s)' % torch.cuda.device_count())


# save directory
model_dir = args.resume_path
log, logclose = create_logger(log_filename=os.path.join(model_dir, 'test.log'))


# load the data
data_path = args.data_path
train_dir = data_path + 'train/'
valid_dir = data_path + 'valid/'
test_dir = data_path + 'test/'
OOD_dir = data_path + 'OOD/train/'
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
test_batch_size = args.batch_size
train_push_batch_size = args.batch_size

# dataset setup

transform_test = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])


# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transform_test)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=False)
# valid set
valid_dataset = datasets.ImageFolder(
    valid_dir,
    transform_test)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=valid_batch_size, shuffle=False)    
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transform_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=valid_batch_size, shuffle=False)
# OOD set
OOD_dataset = datasets.ImageFolder(
    OOD_dir,
    transform_test)
OOD_loader = torch.utils.data.DataLoader(
    OOD_dataset, batch_size=valid_batch_size, shuffle=False)

log('training set size: {0}'.format(len(train_loader.dataset)))
log('valid set size: {0}'.format(len(valid_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))		
log('batch size: {0}'.format(train_batch_size))


# construct the tree
root = Node("root")
root.add_children(['animal','vehicle','everyday_object','weapon','scuba_diver'])
root.add_children_to('animal',['non_primate','primate'])
root.add_children_to('non_primate',['African_elephant','giant_panda','lion'])
root.add_children_to('primate',['capuchin','gibbon','orangutan'])
root.add_children_to('vehicle',['ambulance','pickup','sports_car'])
root.add_children_to('everyday_object',['laptop','sandal','wine_bottle'])
root.add_children_to('weapon',['assault_rifle','rifle'])
# root.add_children(['scuba_diver','African_elephant','giant_panda','lion','capuchin','gibbon','orangutan','ambulance','pickup','sports_car','laptop','sandal','wine_bottle','assault_rifle','rifle'])
root.assign_all_descendents()

OODroot = Node("root")
OODroot.add_children(['animal','vehicle','everyday_object','weapon','scuba_diver'])
OODroot.add_children_to('animal',['non_primate','primate'])
OODroot.add_children_to('non_primate',['king_penguin','tree_frog','zebra'])
OODroot.add_children_to('primate',['macaque','gorilla','chimpanzee'])
OODroot.add_children_to('vehicle',['cab','forklift','tractor','mountain_bike'])
OODroot.add_children_to('everyday_object',['golf_ball','wallet','table_lamp'])
OODroot.add_children_to('weapon',['revolver','bow'])
OODroot.assign_all_descendents()



# construct the model
pretrained = False
resume_path = os.path.join(args.resume_path,"best_model_full_opt.pth")
vgg = model.vgg16(pretrained = pretrained, num_classes=args.num_classes, latent_dim=args.latent_dim, resume = args.resume_path is not None, path = resume_path)
vgg = vgg.cuda()
vgg_multi = torch.nn.DataParallel(vgg)
class_specific = True


# dictionaries

class_names = os.listdir(train_dir)
class_names.sort()
label2name = {i : name for (i,name) in enumerate(class_names)}

OODclass_names = os.listdir(OOD_dir)
OODclass_names.sort()
OODlabel2name = {i : name for (i,name) in enumerate(OODclass_names)}

#test
# log('train')
# train_acc = tnt.test(model=vgg_multi, dataloader=train_loader, root=root, label2name=label2name, class_specific=class_specific, log=log)

log('valid')
valid_acc = tnt.test(model=vgg_multi, dataloader=valid_loader, root=root, label2name=label2name, class_specific=class_specific, log=log)

log('test')
test_acc = tnt.test(model=vgg_multi, dataloader=test_loader, root=root, label2name=label2name, class_specific=class_specific, log=log)

log('OOD')
OOD_acc = tnt.OOD_test(model=vgg_multi, dataloader=OOD_loader, label2name=OODlabel2name, IDroot=root, OODroot = OODroot, class_specific=class_specific, log=log)

log('\nfine accs:')
# log("\ttrain acc: %.2f" % (train_acc * 100))
log("valid acc: %.2f" % (valid_acc * 100))
log("test acc: %.2f" % (test_acc * 100))
log("OOD (coarse) acc: %.2f" % (OOD_acc * 100))

logclose()
