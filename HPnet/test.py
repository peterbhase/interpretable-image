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
import numpy as np


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
parser.add_argument('--data_path', default="../datasets/imagenet/", type=str)
parser.add_argument('--model_path', default=None, type=str,
                            help = "Path for pretrained base model")
parser.add_argument('--resume_path', default=None, type=str,
                            help = "Path for model to resume training with") # to the dir, not .pth
parser.add_argument('--model_name', default="last", type=str)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--n_protos_per_class', default=8, type=int)
parser.add_argument('--batch_mult', default=1, type=int)
parser.add_argument('--proto_dim', default=32, type=int)
parser.add_argument('--KNN_stats', default=True, type=bool)
parser.add_argument('--mask', default=False, type=bool)


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
OOD_dir = data_path + 'OODall/train'
train_push_dir = train_dir
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
test_batch_size = args.batch_size
train_push_batch_size = args.batch_size
nearest_train = os.path.join(model_dir,"nearest_train")
nearest_test = os.path.join(model_dir,"nearest_test")

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
root.assign_all_descendents()

flat_root = Node("root")
flat_root.add_children(['scuba_diver','African_elephant','giant_panda','lion','capuchin','gibbon','orangutan','ambulance','pickup','sports_car','laptop','sandal','wine_bottle','assault_rifle','rifle'])
flat_root.assign_all_descendents()

OODroot = Node("root")
OODroot.add_children(['animal','vehicle','everyday_object','weapon',"scuba_diver"])
OODroot.add_children_to('animal',['non_primate','primate'])
OODroot.add_children_to('non_primate',['king_penguin','tree_frog','zebra'])
OODroot.add_children_to('primate',['macaque','gorilla','chimpanzee'])
OODroot.add_children_to('vehicle',['cab','forklift','tractor','mountain_bike'])
OODroot.add_children_to('everyday_object',['golf_ball','wallet','table_lamp'])
OODroot.add_children_to('weapon',['revolver','bow'])
OODroot.assign_all_descendents()


def KNN_stats(model,train_dir,test_dir):
	name = "root"
	K = 5
	num_protos_per_class = model.module.num_prototypes_per_class
	root = model.module.root
	node = root.get_node(name)

	parent_names = [node.name for node in root.nodes_with_children()]
	parent2stat_train = {name : 0 for name in parent_names}
	parent2stat_test = {name : 0 for name in parent_names}


	for node in root.nodes_with_children():
	    name = node.name
	    n_protos = 6 * node.num_children()
	    
	    children_names = [node.name for node in node.children]
	    
	    for j in range(n_protos):                 
	    
	        train_dir = os.path.join(nearest_train,name,str(j))
	        test_dir = os.path.join(nearest_test,name,str(j))

	        train_class_idx = np.load(os.path.join(train_dir,"class_id.npy"))
	        test_class_idx = np.load(os.path.join(test_dir,"class_id.npy"))
	                
	        train_names = [label2name[idx] for idx in train_class_idx[1:]] # dont count the prototype itself
	        test_names = [label2name[idx] for idx in test_class_idx]
	                
	        proto_name = children_names[j//num_protos_per_class]                
	        proto_node = root.get_node(proto_name)     
	        
	                        
	        if proto_node.has_logits():
	            train_correct = np.array([name in proto_node.descendents for name in train_names])
	            test_correct = np.array([name in proto_node.descendents for name in test_names])
	        else: # leaf node
	            train_correct = np.array([name == proto_name for name in train_names])        
	            test_correct = np.array([name == proto_name for name in test_names])        
	        
	        parent2stat_train[name] += np.mean(train_correct) / n_protos
	        parent2stat_test[name] += np.mean(test_correct) / n_protos

	        
	parent2stat_train = {name : np.round(x,2) for name, x in parent2stat_train.items()}
	parent2stat_test = {name : np.round(x,2) for name, x in parent2stat_test.items()}        

	train_overall = np.mean([y for y in parent2stat_train.values()])
	test_overall = np.mean([y for y in parent2stat_test.values()])

	return train_overall, test_overall



# construct the model
resume_path = os.path.join(args.resume_path,"best_model_" + args.model_name + "_opt.pth")

vgg = model.vgg16_proto(root, pretrained=True, num_prototypes_per_class=args.n_protos_per_class, prototype_dimension=args.proto_dim, img_size=img_size, model_path=args.model_path, 
	resume_path = resume_path)
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


# test
# log('train')
# train_acc = tnt.test(model=vgg_multi, dataloader=train_loader, label2name=label2name, class_specific=class_specific, log=log)

log('valid')
valid_acc, _ = tnt.test(model=vgg_multi, dataloader=valid_loader, label2name=label2name, args=args, class_specific=class_specific, log=log, class_acc = False)

log('test')
test_acc, test_coarse_acc = tnt.test(model=vgg_multi, dataloader=test_loader, label2name=label2name, args=args, class_specific=class_specific, log=log, class_acc = False)

log('OOD')
OOD_acc = tnt.OOD_test(model=vgg_multi, dataloader=OOD_loader, label2name=OODlabel2name, args=args, IDroot=root, OODroot = OODroot, class_specific=class_specific, log=log)

train_KNN, test_KNN = KNN_stats(vgg_multi,train_dir,test_dir)

# log('\nfine accs:')
# log("\ttrain acc: %.2f" % (train_acc * 100))
log("\nvalid acc: %.2f" % (valid_acc * 100))
log("test acc: %.2f" % (test_acc * 100))
log("test coarse acc: %.2f" % (test_coarse_acc * 100))
log("OOD (coarse) acc: %.2f" % (OOD_acc * 100))
log('train KNN: %.3f' % train_KNN)
log('test KNN: %.3f' % test_KNN)


logclose()
