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

if __name__ == "__main__":



	# arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpuid', nargs=1, type=str, default='0')
	parser.add_argument('--data_path', default="../datasets/imagenet/", type=str)
	parser.add_argument('--model_path', default="../model_vgg/saved_models_CEDA1/best_model_full_opt.pth", type=str,
							help = "Path for pretrained base model")
	parser.add_argument('--resume_path', default=None, type=str,
							help = "Path for model to resume training with")
	parser.add_argument('--save_path', default="/usr/xtmp/peterhas/saved_models/", type=str)
	parser.add_argument('--batch_size', default=25, type=int)
	parser.add_argument('--optim', default='adam', type=str)
	parser.add_argument('--CEDA', default=True, type=bool, help = "CEDA data augmentation")
	parser.add_argument('--decay', default=20, type=int)
	parser.add_argument('--push_every', default=5, type=int)
	parser.add_argument('--n_protos_per_class', default=8, type=int)
	parser.add_argument('--batch_mult', default=1, type=int)
	parser.add_argument('--proto_dim', default=32, type=int)
	parser.add_argument('--last_only', default=False, type=bool)
	# HPnet hyperparameters
	parser.add_argument('--lambda_sep', default=.06, type=float)
	parser.add_argument('--lambda_cluster', default=.001, type=float)
	# Pnet hyperparameters
	# parser.add_argument('--lambda_sep', default=1.3, type=float)
	# parser.add_argument('--lambda_cluster', default=.005, type=float)
	# parser.add_argument('--mask', default=True, type=bool)

	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

	start_time = time.time()

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

	# preprocessing
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
	    num_workers=4, pin_memory=False)
	# push set
	train_push_dataset = datasets.ImageFolder(
	    train_push_dir,
	    transform_push)
	train_push_loader = torch.utils.data.DataLoader(
	    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
	    num_workers=4, pin_memory=False)
	# valid set
	valid_dataset = datasets.ImageFolder(
	    valid_dir,
	    transform_test)
	valid_loader = torch.utils.data.DataLoader(
	    valid_dataset, batch_size=valid_batch_size, shuffle=False,
	    num_workers=4, pin_memory=False)

	log('training set size: {0}'.format(len(train_loader.dataset)))
	log('push set size: {0}'.format(len(train_push_loader.dataset)))
	log('valid set size: {0}'.format(len(valid_loader.dataset)))	
	log('effective batch size: {0}'.format(train_batch_size * args.batch_mult))


	# construct the tree -- a bug will occur if tree is too deep (should be resolved if node.unwrap_names_of_joint is re-written)
	root = Node("root")
	root.add_children(['animal','vehicle','everyday_object','weapon','scuba_diver'])
	root.add_children_to('animal',['non_primate','primate'])
	root.add_children_to('non_primate',['African_elephant','giant_panda','lion'])
	root.add_children_to('primate',['capuchin','gibbon','orangutan'])
	root.add_children_to('vehicle',['ambulance','pickup','sports_car'])
	root.add_children_to('everyday_object',['laptop','sandal','wine_bottle'])
	root.add_children_to('weapon',['assault_rifle','rifle'])
	# flat root
	# root.add_children(['scuba_diver','African_elephant','giant_panda','lion','capuchin','gibbon','orangutan','ambulance','pickup','sports_car','laptop','sandal','wine_bottle','assault_rifle','rifle'])
	root.assign_all_descendents()


	# prototype directories
	img_dirs = [os.path.join(model_dir, name + "_prototypes") for name in root.classes_with_children()]
	for img_dir in img_dirs:
	    makedir(img_dir)
	prototype_img_filename_prefix = 'prototype-img'
	prototype_original_img_filename_prefix = 'prototype-original-img'
	proto_bound_boxes_filename_prefix = 'bb'


	# construct the model
	vgg = model.vgg16_proto(root, pretrained=True, num_prototypes_per_class=args.n_protos_per_class, prototype_dimension=args.proto_dim,
		img_size=img_size, model_path=args.model_path, resume_path = args.resume_path)
	vgg = vgg.cuda()
	vgg_multi = torch.nn.DataParallel(vgg)
	class_specific = True

	log('latent space dim: {0}'.format(args.proto_dim))
	log('number of prototypes: {0}'.format(args.n_protos_per_class * sum([node.num_children() for node in vgg.root.nodes_with_children()])))
	log('cluster cost coeff: %.3f' % args.lambda_cluster)
	log('sep cost coeff: %.3f' % args.lambda_sep)

	# define optimizers. this includes optimizers for different parameter groups
	through_protos_optimizer_specs = \
	[{'params': vgg.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
	 {'params': vgg.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
	 {'params': vgg.root_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.animal_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.vehicle_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.everyday_object_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.weapon_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.primate_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.non_primate_prototype_vectors, 'lr': 3e-3}, 
	]
	through_protos_optimizer = torch.optim.Adam(through_protos_optimizer_specs)

	warm_optimizer_specs = \
	[{'params': vgg.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
	 {'params': vgg.root_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.animal_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.vehicle_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.everyday_object_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.weapon_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.primate_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.non_primate_prototype_vectors, 'lr': 3e-3},
	]
	warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

	last_layers_specs = \
	[{'params': vgg.root_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.animal_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.vehicle_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.everyday_object_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.weapon_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.primate_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.non_primate_layer.parameters(), 'lr':1e-3},
	]
	last_layers_optimizer = torch.optim.SGD(last_layers_specs,momentum=.9)

	joint_optimizer_specs = \
	[{'params': vgg.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
	 {'params': vgg.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
	 {'params': vgg.root_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.animal_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.vehicle_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.everyday_object_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.weapon_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.primate_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.non_primate_prototype_vectors, 'lr': 3e-3},
	 {'params': vgg.root_layer.parameters(), 'lr':3e-3},
	 {'params': vgg.animal_layer.parameters(), 'lr':3e-3},
	 {'params': vgg.vehicle_layer.parameters(), 'lr':3e-3},
	 {'params': vgg.everyday_object_layer.parameters(), 'lr':3e-3},
	 {'params': vgg.weapon_layer.parameters(), 'lr':3e-3},
	 {'params': vgg.primate_layer.parameters(), 'lr':3e-3},
	 {'params': vgg.non_primate_layer.parameters(), 'lr':3e-3},
	]
	joint_optimizer = torch.optim.Adam(joint_optimizer_specs)


	optimizers = [through_protos_optimizer,warm_optimizer,joint_optimizer]

	# epochs
	warm_opt = 1
	through_proto_opt = 5 # optimize feature extractor and prototype vectors, but not last layer(s) / classifiers
	joint_opt = 40 # optimize all layers
	last_opt = 5 # optimize only last layers (classifiers)

	best_acc = 0
	best_epoch = 0

	# dictionaries
	class_names = os.listdir(train_dir)
	class_names.sort()
	label2name = {i : name for (i,name) in enumerate(class_names)}
	IDcoarse_names = root.children_names()

	# train the model
	log('start training')

	acc, _ = tnt.valid(model=vgg_multi, dataloader=valid_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

	for epoch in range(warm_opt):
		log('epoch: \t{0}'.format(epoch))    	
		
		tnt.coarse_warm(model=vgg_multi, log=log)
		_ = tnt.train(model=vgg_multi, dataloader=train_loader, label2name=label2name, optimizer=warm_optimizer, args = args, class_specific=class_specific, log=log, warm_up = True)	


	log('optimize through protos')
	for epoch in range(through_proto_opt):
		log('epoch: \t{0}'.format(epoch))    

		# train
		tnt.up_to_protos(model=vgg_multi, log=log)
		_ = tnt.train(model=vgg_multi, dataloader=train_loader, label2name=label2name, optimizer=through_protos_optimizer, args = args, class_specific=class_specific, log=log)
				
		if epoch > 0 and epoch % args.push_every == 0 or epoch == through_proto_opt-1:
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

			acc, _ = tnt.valid(model=vgg_multi, dataloader=valid_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

			save.save_model_w_condition(model=vgg, model_dir=model_dir, model_name='best_model_protos_opt', accu=acc,
						target_accu=best_acc, log=log)

			is_best = acc > best_acc
			best_acc = max(acc, best_acc)
			if is_best:
				best_epoch = epoch			

			tnt.last_layers(model=vgg_multi, log=log)
			_ = tnt.train(model=vgg_multi, dataloader=train_loader, label2name=label2name, optimizer=last_layers_optimizer, args = args, class_specific=class_specific, log=log)

			acc, _ = tnt.valid(model=vgg_multi, dataloader=valid_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

			save.save_model_w_condition(model=vgg, model_dir=model_dir, model_name='best_model_protos_opt', accu=acc,
						target_accu=best_acc, log=log)

			is_best = acc > best_acc
			best_acc = max(acc, best_acc)
			if is_best:
				best_epoch = epoch	
		

		if (epoch+1) % args.decay == 0:
			log('lowered lrs by factor of 10')
			adjust_learning_rate(optimizers)



	best_acc1 = best_acc
	best_epoch1 = best_epoch

	log("optimize joint")
	for epoch in range(joint_opt):

		log('epoch: \t{0}'.format(epoch))

		# layer = getattr(vgg,"root_layer")
		# weights = [p.data for p in layer.parameters()][0]
		# weights = np.array([[np.round(weight.item(),2) for weight in beta] for beta in weights])
		# print("root weights")
		# print(weights)	

		tnt.joint(model=vgg_multi, log=log)
		_ = tnt.train(model=vgg_multi, dataloader=train_loader, label2name=label2name, optimizer=joint_optimizer, args = args, class_specific=class_specific, log=log)				
		
		if epoch > 0 and epoch % args.push_every == 0 or epoch == joint_opt - 1:

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

			acc, _ = tnt.valid(model=vgg_multi, dataloader=valid_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

			save.save_model_w_condition(model=vgg, model_dir=model_dir, model_name='best_model_joint_opt', accu=acc,
						target_accu=best_acc, log=log)

			is_best = acc > best_acc
			best_acc = max(acc, best_acc)
			if is_best:
				best_epoch = epoch	

			for i in range(2):		

				tnt.last_layers(model=vgg_multi, log=log)
				_ = tnt.train(model=vgg_multi, dataloader=train_loader, label2name=label2name, optimizer=last_layers_optimizer, args = args, class_specific=class_specific, log=log)			

				acc, _ = tnt.valid(model=vgg_multi, dataloader=valid_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

				save.save_model_w_condition(model=vgg, model_dir=model_dir, model_name='best_model_joint_opt', accu=acc,
							target_accu=best_acc, log=log)

				is_best = acc > best_acc
				best_acc = max(acc, best_acc)
				if is_best:
					best_epoch = epoch	

		if (epoch+1) % args.decay == 0:
			log('lowered lrs by factor of 10')
			adjust_learning_rate(optimizers)


	best_acc2 = best_acc
	best_epoch2 = best_epoch


	# grab the best model for optimizing last layer
	if best_acc2 > best_acc1:
		resume_path = os.path.join(args.save_path,"best_model_joint_opt.pth")
	else:
		resume_path = os.path.join(args.save_path,"best_model_protos_opt.pth")

	vgg = model.vgg16_proto(root, pretrained=True, num_prototypes_per_class=args.n_protos_per_class, prototype_dimension=args.proto_dim, img_size=img_size, model_path=args.model_path, 
		resume_path = resume_path)
	vgg = vgg.cuda()
	vgg_multi = torch.nn.DataParallel(vgg)


	last_layers_specs = \
	[{'params': vgg.root_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.animal_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.vehicle_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.everyday_object_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.weapon_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.primate_layer.parameters(), 'lr':1e-3},
	 {'params': vgg.non_primate_layer.parameters(), 'lr':1e-3},
	]
	last_layers_optimizer = torch.optim.SGD(last_layers_specs,momentum=.9)

	# always save a copy to start
	acc, _ = tnt.valid(model=vgg_multi, dataloader=valid_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)
	save.save_model_w_condition(model=vgg, model_dir=model_dir, model_name='best_model_last_opt', accu=acc,
	                target_accu=0, log=log)


	# one last push so the prototype images correspond to the right model
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


	log("optimize last layer")
	for epoch in range(last_opt):	
		log('epoch: \t{0}'.format(epoch))	
		
		tnt.last_layers(model=vgg_multi, log=log)
		_ = tnt.train(model=vgg_multi, dataloader=train_loader, label2name=label2name, optimizer=last_layers_optimizer, args = args, class_specific=class_specific, log=log)

		acc, _ = tnt.valid(model=vgg_multi, dataloader=valid_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

		save.save_model_w_condition(model=vgg, model_dir=model_dir, model_name='best_model_last_opt', accu=acc,
		                target_accu=best_acc, log=log)

		is_best = acc > best_acc
		best_acc = max(acc, best_acc)
		if is_best:
			best_epoch = epoch
				

	best_acc3 = best_acc
	best_epoch3 = best_epoch



	log('\n')
	log("up_to_protos acc: %.2f" % (best_acc1 * 100))
	log("occured at epoch: %.2d" % best_epoch1)
	log("joint opt acc: %.2f" %  (best_acc2 * 100))
	log("occured at epoch: %.2d" % best_epoch2)
	log("last only opt acc: %.2f" % (best_acc3 * 100))
	log("occured at epoch: %.2d" % best_epoch3)

	run_time = time.time() - start_time
	log("\ntotal running time: %.3f hours" % (run_time / 3600))

	logclose()
