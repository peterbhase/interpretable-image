import time
import torch
import numpy as np
import os

from helpers import list_of_distances, make_one_hot, joint_distribution
from preprocess import mean, std 
from torchvision.transforms.functional import normalize

def CE(logits,target):
     #manual definition of the cross entropy for a target which is a probability distribution    
     probs = torch.nn.functional.softmax(logits,1)    
     return torch.sum(torch.sum(- target * torch.log(probs))) 


def _train_or_test(model, dataloader, root, label2name, optimizer=None, class_specific=False, log=print, batch_multiplier = 1, CEDA = False, class_acc = False):
    #'''
    #model: the multi-gpu model
    #dataloader:
    #optimizer: if None, will be test evaluation
    #'''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_coarse_correct = 0
    n_fine_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_noise_cross_ent = 0
    total_cluster_cost = 0
    total_separation_cost = 0        
        
    coarse_names = root.children_names()
    num_coarse = len(coarse_names)
    coarse_class_correct = np.array([0 for i in range(root.num_children())])
    coarse_class_total = np.array([0 for i in range(root.num_children())])        

    fine_names = sorted([x for x in label2name.values()])
    num_fine = len(fine_names)
    fine_class_correct = np.array([0 for i in range(len(fine_names))])
    fine_class_total = np.array([0 for i in range(len(fine_names))])

    fineLabel2coarseLabel = {label : root.children_to_labels[root.closest_descendent_for(name).name] for label, name in enumerate(fine_names)}

    # torch.enable_grad() has no effect outside of no_grad()
    grad_req = torch.enable_grad() if is_train else torch.no_grad()
    if not is_train: model.eval()

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        batch_names = [label2name[y.item()] for y in label] 
        batch_size = len(target)
        

        root_y = torch.tensor([fineLabel2coarseLabel[y.item()] for y in label])

        with grad_req:
            
            logits = model(input)           
            
            cross_entropy = torch.nn.functional.cross_entropy(logits,target)
                        
            # evaluation statistics
            probs = torch.nn.functional.softmax(logits.data,1)                        
            
            coarse_probs = torch.zeros(batch_size,num_coarse)
            for j in range(len(label)):
                for k in range(num_fine):
                    coarse_id = fineLabel2coarseLabel[k]
                    coarse_probs[j,coarse_id] += probs[j,k].item()

            fine_preds = torch.argmax(probs,1)
            coarse_preds = torch.argmax(coarse_probs,1)                 

            n_examples += target.size(0)            
            
            fine_correct = fine_preds==target
            n_fine_correct += fine_correct.sum().item()
            coarse_correct = coarse_preds == root_y
            n_coarse_correct += coarse_correct.sum().item()

            if class_acc:
                for j in range(len(target)):
                    coarse_class_correct[root_y[j]] += (1 if coarse_correct[j] else 0)
                    coarse_class_total[root_y[j]] += 1
                    fine_class_correct[target[j]] += (1 if fine_correct[j] else 0)
                    fine_class_total[target[j]] += 1
            
            
            # encourage uniform prediction on noise -- note for batch_size, this takes like .03 seconds to do
            if CEDA:                
                noise = torch.stack([normalize(torch.rand((3,32,32)),mean,std) for n in range(batch_size)]).cuda()    
                unif = torch.ones((batch_size,model.module.num_classes)).cuda() / model.module.num_classes	                           
                
                nlogits = model(input) # no need to call .forward            
                noise_cross_ent = CE(nlogits, unif)   


            n_batches += 1
            total_cross_entropy += cross_entropy.item() 
 

        # compute gradient and do SGD step
        if is_train:
            loss = (cross_entropy + (noise_cross_ent / 20 if CEDA else 0)) / batch_multiplier
            loss.backward()
            if (i+1) % batch_multiplier == 0: # every batch_multiplier steps
                optimizer.step()
                optimizer.zero_grad()


            
        del input, image
        del target, label, root_y
        del logits
        if CEDA: del noise, nlogits, unif
        del probs, coarse_probs
        del fine_correct, coarse_correct
        del fine_preds, coarse_preds


    end = time.time()


    log('\ttime: \t{0:.2f}'.format(end -  start))
    log('\tcross ent: \t{0:.2f}'.format(total_cross_entropy / n_batches))
    if CEDA: log('\tnoise cross ent: \t{0:.2f}'.format(total_noise_cross_ent / n_batches))
    log('\tcoarse acc: \t{0:.2f}%'.format(n_coarse_correct / n_examples * 100))
    log('\tfine acc: \t{0:.2f}%'.format(n_fine_correct / n_examples * 100))    
    #log('\tproto cluster: \t{0:.2f}'.format(proto_cluster_cost.item()))
    #p = model.module.prototype_vectors.view(model.module.num_prototypes, -1)
    #log('\tp dist pair: \t{0}'.format(torch.mean(list_of_distances(p, p)).item()))

    if class_acc:
        log('\n')
        for c in range(len(coarse_names)):
            log('\tAccuracy of %5s : %2d %%' % (coarse_names[c], 100 * coarse_class_correct[c] / coarse_class_total[c]))    
        log('\n')
        for c in range(len(fine_names)):
            log('\tAccuracy of %5s : %2d %%' % (fine_names[c], 100 * fine_class_correct[c] / fine_class_total[c]))            

    return n_fine_correct / n_examples



def _OOD_test(model, dataloader, label2name, IDroot, OODroot, optimizer=None, class_specific=False, log=print, batch_multiplier = 1, CEDA = False, class_acc = False):
    #'''
    #model: the multi-gpu model
    #dataloader:
    #optimizer: if None, will be test evaluation
    #'''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_coarse_correct = 0
    n_fine_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_noise_cross_ent = 0
    total_cluster_cost = 0
    total_separation_cost = 0    
        
    coarse_names = IDroot.children_names()
    num_coarse = len(coarse_names)
    coarse_class_correct = np.array([0 for i in range(num_coarse)])
    coarse_class_total = np.array([0 for i in range(num_coarse)])        

    fine_names = sorted([x for x in label2name.values()])    
    fine_class_correct = np.array([0 for i in range(len(fine_names))])
    fine_class_total = np.array([0 for i in range(len(fine_names))])
    fineLabel2coarseLabel = {label : OODroot.children_to_labels[OODroot.closest_descendent_for(name).name] for label, name in enumerate(fine_names)}
    
    IDfine_names = [node.name for node in IDroot.nodes_without_children()]
    IDfine_names.sort()
    num_fine = (len(IDfine_names))
    IDfineLabel2coarseLabel = {label : IDroot.children_to_labels[IDroot.closest_descendent_for(name).name] for label, name in enumerate(IDfine_names)}    

    
    # torch.enable_grad() has no effect outside of no_grad()
    grad_req = torch.enable_grad() if is_train else torch.no_grad()
    if not is_train: model.eval()

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        batch_names = [label2name[y.item()] for y in label] 
        batch_size = len(target)        

        root_y = torch.tensor([fineLabel2coarseLabel[y.item()] for y in label])

        with grad_req:
            
            logits = model(input)           
            
            cross_entropy = torch.nn.functional.cross_entropy(logits,target)
                        
            # evaluation statistics
            probs = torch.nn.functional.softmax(logits.data,1)                        
            
            coarse_probs = torch.zeros(batch_size,num_coarse)
            for j in range(len(label)):
                for k in range(num_fine):
                    coarse_id = IDfineLabel2coarseLabel[k]
                    coarse_probs[j,coarse_id] += probs[j,k].item()
            coarse_preds = torch.argmax(coarse_probs,1)                         
            coarse_correct = coarse_preds == root_y

            fine_preds = torch.argmax(probs,1)

                
            # check if fine prediction is in correct group
            # coarse_correct = torch.zeros(batch_size)
            # for j in range(len(target)):
            #     fine_pred = fine_preds[j]
            #     coarse_target = root_y[j]
            #     if IDfine_names[fine_pred] in IDroot.children[coarse_target].descendents:
            #         coarse_correct[j] += 1
                

            n_examples += target.size(0)                                    
            n_coarse_correct += coarse_correct.sum().item()

            if class_acc:
                for j in range(batch_size):
                    coarse_class_correct[root_y[j]] += (1 if coarse_correct[j] else 0)
                    coarse_class_total[root_y[j]] += 1
                    fine_class_correct[target[j]] += (1 if coarse_correct[j] else 0)
                    fine_class_total[target[j]] += 1        


            n_batches += 1
            total_cross_entropy += cross_entropy.item() 
            
        del input
        del target, root_y
        del logits
        del probs, coarse_probs
        del fine_preds, coarse_preds
        del coarse_correct
        
    end = time.time()

    log('\ttime: \t{0:.2f}'.format(end -  start))
    log('\tcross ent: \t{0:.2f}'.format(total_cross_entropy / n_batches))
    if CEDA: log('\tnoise cross ent: \t{0:.2f}'.format(total_noise_cross_ent / n_batches))
    log('\tcoarse acc: \t{0:.2f}%'.format(n_coarse_correct / n_examples * 100))
    log('\tfine acc: \t{0:.2f}%'.format(n_fine_correct / n_examples * 100))    
    #log('\tproto cluster: \t{0:.2f}'.format(proto_cluster_cost.item()))
    #p = model.module.prototype_vectors.view(model.module.num_prototypes, -1)
    #log('\tp dist pair: \t{0}'.format(torch.mean(list_of_distances(p, p)).item()))

    if class_acc:
        log('\n')
        for c in range(len(coarse_names)):
            if coarse_names[c] != "scuba_diver":
                log('\tAccuracy of %5s : %2d %%' % (coarse_names[c], 100 * coarse_class_correct[c] / coarse_class_total[c]))    
            else:
                print("\tskipping scuba diver")   
        log('\n')
        for c in range(len(fine_names)):
            log('\tAccuracy of %5s : %2d %%' % (fine_names[c], 100 * fine_class_correct[c] / fine_class_total[c]))  

    return n_coarse_correct / n_examples





def train(model, dataloader, optimizer, root, label2name,  class_specific=False, log=print, batch_multiplier = 1, CEDA = False):
    assert(optimizer is not None)
    log('train')
    return _train_or_test(model=model, dataloader=dataloader, label2name = label2name, optimizer=optimizer, root=root, class_specific=class_specific, log=log,
         batch_multiplier = batch_multiplier, CEDA=CEDA)

def valid(model, dataloader, root, label2name,  class_specific=False, log=print):
    log('valid')
    return _train_or_test(model=model, dataloader=dataloader, label2name = label2name, optimizer=None, root=root, class_specific=class_specific, log=log, class_acc = False)    

def test(model, dataloader, root, label2name, class_specific=False, log=print):
    #log('test')
    return _train_or_test(model=model, dataloader=dataloader, label2name = label2name, optimizer=None, root=root, class_specific=class_specific, log=log, class_acc = True)

def OOD_test(model, dataloader, label2name, IDroot, OODroot, class_specific=False, log=print):
    #log('test')
    return _OOD_test(model=model, dataloader=dataloader, label2name = label2name, optimizer=None, IDroot=IDroot, OODroot=OODroot, class_specific=class_specific, log=log, class_acc = True)