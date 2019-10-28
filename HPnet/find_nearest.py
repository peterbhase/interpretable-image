#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:35:57 2018

@author: cfchen
"""

import torch
import numpy as np

import heapq

import matplotlib.pyplot as plt
import os
import copy
import time

import cv2

from receptive_field import compute_rf_prototype
from helpers import makedir

class ImagePatch:
    
    def __init__(self, patch, label, distance,
                 original_img=None, act_pattern=None, patch_indices=None):
        self.patch = patch
        self.label = label
        self.negative_distance = -distance
        
        self.original_img = original_img
        self.act_pattern = act_pattern
        self.patch_indices = patch_indices
    
    def __lt__(self, other):
        return self.negative_distance < other.negative_distance

# find the nearest patches in the dataset to each prototype
def find_k_nearest_patches_to_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                                         prototype_network_parallel, # pytorch network with prototype_vectors
                                         k=5,
                                         preprocess_input_function=None, # normalize if needed)
                                         prototype_layer_stride=1,
                                         root_dir_for_saving_images='./nearest_',
                                         save_image_class_identity=True,
                                         log=print):
    log('find nearest patches')
    start = time.time()
    
    protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
    parent_names = [node.name for node in prototype_network_parallel.module.root.nodes_with_children()]

    cmap = "jet"
    
    node2heaps = {name : [] for name in parent_names}
    # for each parent node, organize a heap for every prototype
    for node in prototype_network_parallel.module.root.nodes_with_children():
      name = node.name
      num_prototypes = getattr(prototype_network_parallel.module,"num_" + node.name + "_prototypes")
      for j in range(num_prototypes):
        node2heaps[name].append([])
    
    for (search_batch_input, search_y) in dataloader:
        if preprocess_input_function != None:
            #print('preprocessing input for pushing ...')
            search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch)
        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.cuda()            
            conv_features = prototype_network_parallel.module.conv_features(search_batch)        
        

        for node in prototype_network_parallel.module.root.nodes_with_children():

          num_prototypes = getattr(prototype_network_parallel.module,"num_" + node.name + "_prototypes")

          with torch.no_grad():
            proto_dist_torch = prototype_network_parallel.module.prototype_distances(conv_features, node.name)
            proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

          heaps = node2heaps[node.name]
        
          for img_idx, distance_map in enumerate(proto_dist_):
              for j in range(num_prototypes):
                  # find the closest patch to prototype j
                  closest_patch_distance_to_prototype_j = np.amin(distance_map[j])
                  closest_patch_indices_in_distance_map_j = \
                      list(np.unravel_index(np.argmin(distance_map[j],axis=None),
                                            distance_map[j].shape))
                  closest_patch_indices_in_distance_map_j = [0] + closest_patch_indices_in_distance_map_j
                  closest_patch_indices_in_img = \
                      compute_rf_prototype(search_batch.size(2),
                                           closest_patch_indices_in_distance_map_j,
                                           protoL_rf_info)
                  closest_patch = \
                      search_batch_input[img_idx, :,
                                         closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2],
                                         closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]
                  closest_patch = closest_patch.numpy()
                  closest_patch = np.transpose(closest_patch, (1, 2, 0))
                  
                  original_img = search_batch_input[img_idx].numpy()
                  original_img = np.transpose(original_img, (1, 2, 0))
                  
                  act_pattern = np.log(1 + (1/(distance_map[j] + 1e-4)))
                  
                  patch_indices = closest_patch_indices_in_img[1:5]
                  
                  # construct the closest patch object
                  closest_patch = ImagePatch(patch=closest_patch,
                                             label=search_y[img_idx],
                                             distance=closest_patch_distance_to_prototype_j,
                                             original_img=original_img,
                                             act_pattern=act_pattern,
                                             patch_indices=patch_indices)
                  # add to the j-th heap
                  if len(heaps[j]) < k:
                      heapq.heappush(heaps[j], closest_patch)
                  else:
                      heapq.heappushpop(heaps[j], closest_patch)


        del conv_features, search_batch

      
    for node in prototype_network_parallel.module.root.nodes_with_children():
        heaps = node2heaps[node.name]
        num_prototypes = getattr(prototype_network_parallel.module,"num_" + node.name + "_prototypes")

        for j in range(num_prototypes):
            heaps[j].sort()
            heaps[j] = heaps[j][::-1] # reverses
            
            dir_for_saving_images = os.path.join(root_dir_for_saving_images,
                                                 node.name,
                                                 str(j))
            makedir(dir_for_saving_images)
            
            labels = []
            
            for i, patch in enumerate(heaps[j]):
                # save the patch itself
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i) + '.png'),
                           arr=patch.patch,
                           vmin=0.0,
                           vmax=1.0)
                # save the original image where the patch comes from
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i) + '_original.png'),
                           arr=patch.original_img,
                           vmin=0.0,
                           vmax=1.0)
                # save the activation pattern and the patch indices
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i) + '_act.npy'),
                        patch.act_pattern)
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i) + '_indices.npy'),
                        patch.patch_indices)
                # upsample the activation pattern
                img_size = patch.original_img.shape[0]
                original_img_gray = cv2.cvtColor(patch.original_img, cv2.COLOR_BGR2GRAY)
                upsampled_act_pattern = cv2.resize(patch.act_pattern,
                                                   dsize=(img_size, img_size),
                                                   interpolation=cv2.INTER_CUBIC)
                # overlay heatmap on the original image and save the result
                overlayed_original_img = 0.7 * original_img_gray + 0.3 * upsampled_act_pattern                
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i) + '_original_with_heatmap.png'),
                           arr=overlayed_original_img,
                           cmap=cmap,
                           vmin=0.0,
                           vmax=1.0)
                # save the patch with heatmap
                overlayed_patch = overlayed_original_img[patch.patch_indices[0]:patch.patch_indices[1],
                                                         patch.patch_indices[2]:patch.patch_indices[3]]
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i) + '_patch_with_heatmap.png'),
                           arr=overlayed_patch,
                           cmap=cmap,
                           vmin=0.0,
                           vmax=1.0)
            
            if save_image_class_identity:
                labels = np.array([patch.label for patch in heaps[j]])
                np.save(os.path.join(dir_for_saving_images, 'class_id.npy'),
                        labels)
      
    end = time.time()
    log('\tfind nearest patches time: \t{0}'.format(end -  start))
