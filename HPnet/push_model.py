import os
import shutil

import torch
import torch.utils.data
import transforms
import torchvision.datasets as datasets
import argparse
from helpers import makedir, adjust_learning_rate
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, img_size
from node import Node
import time


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
parser.add_argument('--data_path', default="/usr/xtmp/peterhas/datasets/imagenetIDsmall/", type=str)
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--resume_path', default=None, type=str)
parser.add_argument('--save_path', default="/usr/xtmp/peterhas/saved_models/", type=str)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--optim', default='adam', type=str)
parser.add_argument('--CEDA', default=True, type=bool)
parser.add_argument('--decay', default=20, type=int)
parser.add_argument('--push_every', default=5, type=int)
parser.add_argument('--n_protos_per_class', default=6, type=int)
parser.add_argument('--batch_mult', default=1, type=int)
parser.add_argument('--proto_dim', default=32, type=int)
parser.add_argument('--last_only', default=False, type=bool)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
#print('training with %d gpu(s)' % torch.cuda.device_count())


# save directory
model_dir = args.save_path
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))


# load the data
data_path = args.data_path
train_dir = data_path + 'train/'
valid_dir = data_path + 'valid/'
test_dir = data_path + 'test/'
train_push_dir = train_dir
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
test_batch_size = args.batch_size
train_push_batch_size = args.batch_size

# dataset setup
# processing

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])

transform_push = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])


# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transform_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    )#num_workers=4, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transform_push)
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    )#num_workers=4, pin_memory=False)
# valid set
valid_dataset = datasets.ImageFolder(
    valid_dir,
    transform_test)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=valid_batch_size, shuffle=False,
    )#num_workers=4, pin_memory=False)

log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('valid set size: {0}'.format(len(valid_loader.dataset)))	
log('batch size: {0}'.format(train_batch_size))
log('batch mult: {0}'.format(args.batch_mult))


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


# prototype dirs
img_dirs = [os.path.join(model_dir,name + "_prototypes") for name in root.classes_with_children()]
for img_dir in img_dirs:
    makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_original_img_filename_prefix = 'prototype-original-img'
proto_bound_boxes_filename_prefix = 'bb'



# construct the model
vgg = model.vgg16_proto(root, pretrained=True, num_prototypes_per_class=args.n_protos_per_class, prototype_dimension=args.proto_dim,
	img_size=img_size, model_path=args.model_path, resume_path = args.resume_path)
vgg = vgg.cuda()
vgg_multi = torch.nn.DataParallel(vgg)
class_specific = True


# dictionaries

class_names = os.listdir(train_dir)
class_names.sort()
label2name = {i : name for (i,name) in enumerate(class_names)}


# push

acc = tnt.valid(model=vgg_multi, dataloader=valid_loader, label2name=label2name, class_specific=class_specific, log=log)


push.push_prototypes(
	train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
	prototype_network_parallel=vgg_multi, # pytorch network with prototype_vectors
	label2name=label2name,
	class_specific=class_specific,
	preprocess_input_function=preprocess_input_function, # normalize if needed
	prototype_layer_stride=1,           
	root_dir_for_saving_prototypes=model_dir, # if not None, prototypes will be saved here
	prototype_img_filename_prefix=prototype_img_filename_prefix,
	prototype_original_img_filename_prefix=prototype_original_img_filename_prefix,
	proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
	save_prototype_class_identity=True,
	log=log)


acc = tnt.valid(model=vgg_multi, dataloader=valid_loader, label2name=label2name, class_specific=class_specific, log=log)

save.save_model_w_condition(model=vgg, model_dir=model_dir, model_name='best_model_pushed_opt', accu=acc,
	                target_accu=0, log=log)