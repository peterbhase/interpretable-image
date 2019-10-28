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


parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
# Random Erasing
parser.add_argument('--p', default=0, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')
parser.add_argument('--lr', default=.01, type=float, help='learning rate')
parser.add_argument('--decay', default=30, type=int, help='learning rate decays every')
parser.add_argument('--epochs', default=100, type=int, help='num epochs')
parser.add_argument('--resume_path', default=None, type=str, help='')
parser.add_argument('--save_path', default="saved_models", type=str, help='')
parser.add_argument('--data_path', default="../datasets/imagenet/", type=str, help='')
parser.add_argument('--batch_size', default=75, type=int, help='batch size')
parser.add_argument('--num_classes', default=15, type=int, help='')
parser.add_argument('--latent_dim', default=512, type=int, help='')
parser.add_argument('--CEDA', default=True, type=bool)
parser.add_argument('--batch_multiplier', default=1, type=int, help='')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# book keeping namings and code
model_dir = args.save_path
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

if __name__ == "__main__":

	# load the data
	data_path = args.data_path
	train_dir = data_path + 'train/'
	valid_dir = data_path + 'valid/'
	train_push_dir = train_dir
	train_batch_size = args.batch_size
	valid_batch_size = args.batch_size


	# dataset setup

	transform_train = transforms.Compose([
	    transforms.RandomResizedCrop(img_size,scale=(.2,1)),
	    transforms.RandomHorizontalFlip(),    
	    transforms.ToTensor(),
	    transforms.Normalize(mean, std),
	])


	transform_test = transforms.Compose([
	    transforms.Resize(256),
	    transforms.CenterCrop(img_size),
	    transforms.ToTensor(),
	    transforms.Normalize(mean, std),
	])


	# train set
	train_dataset = datasets.ImageFolder(
	    train_dir,
	    transform_train)
	train_loader = torch.utils.data.DataLoader(
	    train_dataset, batch_size=train_batch_size, shuffle=True,
	    num_workers=4, pin_memory=False)    
	# valid set
	valid_dataset = datasets.ImageFolder(
	    valid_dir,
	    transform_test)
	valid_loader = torch.utils.data.DataLoader(
	    valid_dataset, batch_size=valid_batch_size, shuffle=False,
	    num_workers=4, pin_memory=False)
	    

	log('training set size: {0}'.format(len(train_loader.dataset)))
	log('valid set size: {0}'.format(len(valid_loader.dataset)))
	log('batch size: {0}'.format(train_batch_size))
	log('batch multiplier: {0}'.format(args.batch_multiplier))

	# construct the model
	pretrained = False
	vgg = model.vgg16(pretrained = pretrained, num_classes=args.num_classes, latent_dim = args.latent_dim, resume = args.resume_path is not None, path = args.resume_path)
	vgg = vgg.cuda()
	vgg_multi = torch.nn.DataParallel(vgg)
	class_specific = True # deprecated

	# define optimizer
	lr = args.lr
	full_optimizer_specs = \
	[{'params': vgg.features.parameters(), 'lr': lr, 'weight_decay': 5e-4},
	 {'params': vgg.classifier.parameters(), 'lr': lr, 'weight_decay': 5e-4},
	]	
	#full_optimizer = torch.optim.Adam(full_optimizer_specs)
	full_optimizer = torch.optim.SGD(full_optimizer_specs,momentum=.9)

	class_optimizer_specs = [{'params': vgg.classifier.parameters(), 'lr': lr, 'weight_decay': 1e-4}]
	# class_optimizer = torch.optim.Adam(class_optimizer_specs)
	class_optimizer = torch.optim.SGD(class_optimizer_specs,momentum=.9)

	optimizers = (full_optimizer, class_optimizer)

	# epochs

	epochs1 = args.epochs
	epochs2 = 0
	best_acc = 0
	best_epoch = 0


	# construct the tree -- just for computing class-specific 
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

	# name dictionary
	class_names = os.listdir(train_dir)
	class_names.sort()
	label2name = {i : name for (i,name) in enumerate(class_names)}


	# train the model and optionally compute acc
	log('start training')
	# acc = tnt.valid(model=vgg_multi, dataloader=valid_loader, root = root, label2name = label2name, class_specific=class_specific, log=log)


	for epoch in range(epochs1):
		log('epoch: \t{0}'.format(epoch))    
		adjust_learning_rate(optimizers,epoch,lr,args.decay)
				
		_ = tnt.train(model=vgg_multi, dataloader=train_loader, root = root, label2name = label2name, optimizer=full_optimizer, class_specific=class_specific, log=log, 
				batch_multiplier = args.batch_multiplier, CEDA = args.CEDA)

		
		acc = tnt.valid(model=vgg_multi, dataloader=valid_loader, root = root, label2name = label2name, class_specific=class_specific, log=log)

		
		save.save_model_w_condition(model=vgg, model_dir=model_dir, model_name='best_model_full_opt', accu=acc,
		                target_accu=best_acc, log=log)

		is_best = acc > best_acc
		best_acc = max(acc, best_acc)
		if is_best:
			best_epoch = epoch	


	best_acc1 = best_acc
	best_epoch1 = best_epoch

	# log("optimize classifier") -- not necessary for vgg
	# for epoch in range(epochs1,epochs2):

	# 	log('epoch: \t{0}'.format(epoch))    
	# 	adjust_learning_rate(optimizers,epochs1+epoch,lr,args.decay)
				
	# 	_ = tnt.train(model=vgg_multi, dataloader=train_loader, root = root, label2name = label2name, optimizer=class_optimizer, class_specific=class_specific, log=log, CEDA = False)

		
	# 	acc = tnt.valid(model=vgg_multi, dataloader=valid_loader, root = root, label2name = label2name, class_specific=class_specific, log=log)

		
	# 	save.save_model_w_condition(model=vgg, model_dir=model_dir, model_name='best_model_class_opt', accu=acc,
	# 	                target_accu=best_acc, log=log)

	# 	is_best = acc > best_acc
	# 	best_acc = max(acc, best_acc)
	# 	if is_best:
	# 		best_epoch = epoch	


	# best_acc2 = best_acc
	# best_epoch2 = best_epoch


	log('\n')
	log("full opt acc: %.2f" % (best_acc1 * 100))
	log("occured at epoch: %.2d" % best_epoch1)
	# log("class opt acc: %.2f" %  (best_acc2 * 100))
	# log("occured at epoch: %.2d" % best_epoch2)

	logclose()
