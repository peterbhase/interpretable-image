#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:35:56 2018

@author: cfchen
"""
import torch
import numpy as np
import matplotlib.image
import os
import copy
import time
import matplotlib.pyplot as plt

from receptive_field import compute_rf_prototype
from helpers import makedir

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors        
                    label2name,            
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_original_img_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,
                    skip_coarse = False,
                    log=print):

    log('\tpush')
    start = time.time()
      

    proto_epoch_dir = None
    search_batch_size = dataloader.batch_size

    for node in prototype_network_parallel.module.root.nodes_with_children():
        
        n_prototypes = getattr(prototype_network_parallel.module,"num_" + node.name + "_prototypes")
        prototype_shape = getattr(prototype_network_parallel.module,node.name + "_prototype_shape")
        node.global_min_proto_dist = np.full(n_prototypes, np.inf)
        node.global_min_fmap_patches = np.zeros([n_prototypes,
                                            prototype_shape[1]*prototype_shape[2]*prototype_shape[3]])
        node.proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                              fill_value=-1)        


    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        start_index_of_search_batch = push_iter*search_batch_size

        search_batch = copy.deepcopy(search_batch_input)        
        search_batch = preprocess_input_function(search_batch)
        search_batch = search_batch.cuda()
        with torch.no_grad():
            conv_features = prototype_network_parallel.module.conv_features(search_batch)


        

        batch_names = [label2name[y.item()] for y in search_y]
        for node in prototype_network_parallel.module.root.nodes_with_children():
            # get names specific to children
            # print("\t" + node.name)`
            children_idx = torch.tensor([name in node.descendents for name in batch_names])
            batch_names_coarsest = [node.closest_descendent_for(name).name for name in batch_names if name in node.descendents]         
            node_y = torch.tensor([node.children_to_labels[name] for name in batch_names_coarsest])            
             
            
            update_prototypes_on_batch(search_batch_input[children_idx],
                                       start_index_of_search_batch,
                                       prototype_network_parallel,
                                       conv_features,
                                       node.global_min_proto_dist,
                                       node.global_min_fmap_patches,
                                       node.proto_bound_boxes,
                                       name = node.name,
                                       n_prototypes_per_class=node.num_prototypes_per_class,
                                       prototype_shape =node.prototype_shape,
                                       search_y=node_y,
                                       num_classes=node.num_children(),
                                       preprocess_input_function=preprocess_input_function,
                                       prototype_layer_stride=prototype_layer_stride,
                                       dir_for_saving_prototypes=os.path.join(root_dir_for_saving_prototypes,node.name + "_prototypes"),
                                       prototype_img_filename_prefix=prototype_img_filename_prefix,
                                       prototype_original_img_filename_prefix=prototype_original_img_filename_prefix)

      
        del conv_features, search_batch

        
    log('\tExecuting push ...')
    for node in prototype_network_parallel.module.root.nodes_with_children():
        img_dir = os.path.join(root_dir_for_saving_prototypes,node.name + "_prototypes")     
        np.save(os.path.join(img_dir, "bb" + '.npy'), node.proto_bound_boxes)
 
        prototype_update = np.reshape(node.global_min_fmap_patches,tuple(node.prototype_shape))
        prototype_update = torch.tensor(prototype_update, dtype=torch.float32).cuda()
        getattr(prototype_network_parallel.module, node.name + "_prototype_vectors").data.copy_(prototype_update)
            
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))


# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               conv_features,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_bound_boxes,       # this will be updated
                               name,
                               prototype_shape,                               
                               n_prototypes_per_class=None, # default: no restriction on number of prototypes per class
                               search_y=None,    # required if n_prototypes_per_class != None
                               num_classes=None, # required if n_prototypes_per_class != None
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_original_img_filename_prefix=None):


    # print("inside update")    
    # for x in range(5):
    #     img_from_input = np.transpose(search_batch_input[x],(1,2,0))
    #     plt.imshow(img_from_input)
    #     plt.show() 

    protoL_input_torch = copy.deepcopy(conv_features)

    with torch.no_grad():
        
        proto_dist_torch = prototype_network_parallel.module.prototype_distances(conv_features, name)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
  
    del protoL_input_torch, proto_dist_torch

    if n_prototypes_per_class != None:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]

    for j in range(n_prototypes):
        # print("STARTING PROTO", j)
        if n_prototypes_per_class != None:
            target_class = j // n_prototypes_per_class
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            # print(n_prototypes_per_class)
            # print(j)
            # print(target_class)            
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
        else:
            proto_dist_j = proto_dist_[:,j,:,:]
        batch_min_proto_dist_j = np.amin(proto_dist_j)
        
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            # print("batch min index", batch_argmin_proto_dist_j)
            # retrive the corresponding feature map patch
            if n_prototypes_per_class != None:
                img_index_in_batch = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]
            else:
                img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1]*prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2]*prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w
            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]
            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = np.reshape(batch_min_fmap_patch_j, -1)

            if n_prototypes_per_class != None:
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch_input.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            img_j = search_batch_input[rf_prototype_j[0],
                                       :,
                                       rf_prototype_j[1]:rf_prototype_j[2],
                                       rf_prototype_j[3]:rf_prototype_j[4]]
            img_j = img_j.numpy()
            proto_bound_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_bound_boxes[j, 1] = rf_prototype_j[1]
            proto_bound_boxes[j, 2] = rf_prototype_j[2]
            proto_bound_boxes[j, 3] = rf_prototype_j[3]
            proto_bound_boxes[j, 4] = rf_prototype_j[4]
            if proto_bound_boxes.shape[1] == 6 and not(search_y is None):
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes != None:
                if prototype_img_filename_prefix != None:
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_img_filename_prefix + str(j) + '.npy'),
                            img_j)
                    img_j = np.transpose(img_j, (1, 2, 0))
                    matplotlib.image.imsave(os.path.join(dir_for_saving_prototypes,
                                                         prototype_img_filename_prefix + str(j) + '.png'),
                                            img_j,
                                            vmin=0.0,
                                            vmax=1.0)
                if prototype_original_img_filename_prefix != None:
                    original_img_j = search_batch_input[rf_prototype_j[0]]
                    original_img_j = original_img_j.numpy()
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_original_img_filename_prefix + str(j) + '.npy'),
                            original_img_j)
                    original_img_j = np.transpose(original_img_j, (1, 2, 0))               
                    # scipy.misc.imsave(os.path.join(dir_for_saving_prototypes,
                    #                                prototype_original_img_filename_prefix + str(j) + '.png'),
                    #                   original_img_j)
                    matplotlib.image.imsave(os.path.join(dir_for_saving_prototypes,
                                                         prototype_original_img_filename_prefix + str(j) + '.png'),
                                            original_img_j,
                                            vmin=0.0,
                                            vmax=1.0)

    if n_prototypes_per_class != None:
        del class_to_img_index_dict
