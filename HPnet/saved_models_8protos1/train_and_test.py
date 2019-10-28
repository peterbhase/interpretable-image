import time
import torch
import numpy as np

from helpers import list_of_distances, make_one_hot
from preprocess import mean, std 
from torchvision.transforms.functional import normalize
from node import Node

def CE(logits,target):
     #manual definition of the cross entropy for a target which is a probability distribution    
     probs = torch.nn.functional.softmax(logits,1)    
     return torch.sum(torch.sum(- target * torch.log(probs))) 


def _train_or_test(model, dataloader, label2name, optimizer=None, args = None, class_specific=False, log=print, warm_up = False, CEDA = False, batch_mult = 1, class_acc = False):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    if not is_train: model.eval()
    start = time.time()
    n_examples = 0
    n_coarse_correct = 0
    n_fine_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_noise_cross_ent = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_l1 = 0
 
        
    coarse_names = model.module.root.children_names()
    num_coarse = len(coarse_names)
    coarse_class_correct = np.array([0 for i in range(model.module.root.num_children())])
    coarse_class_total = np.array([0 for i in range(model.module.root.num_children())])

    fine_names = [x for x in label2name.values()]
    num_fine = len(fine_names)
    fine_class_correct = np.array([0 for i in range(len(fine_names))])
    fine_class_total = np.array([0 for i in range(len(fine_names))])

    num_parents = len([node for node in model.module.root.nodes_with_children()])

    model.module.root.assign_unif_distributions()

    fineLabel2coarseLabel = {label : model.module.root.children_to_labels[model.module.root.closest_descendent_for(name).name] for label, name in enumerate(fine_names)} 

    flat_model = not model.module.root.children[0].has_logits()          
    if flat_model: 
        print("using flat model")
        root = Node("root")
        root.add_children(['animal','vehicle','everyday_object','weapon','scuba_diver'])
        root.add_children_to('animal',['non_primate','primate'])
        root.add_children_to('non_primate',['African_elephant','giant_panda','lion'])
        root.add_children_to('primate',['capuchin','gibbon','orangutan'])
        root.add_children_to('vehicle',['ambulance','pickup','sports_car'])
        root.add_children_to('everyday_object',['laptop','sandal','wine_bottle'])
        root.add_children_to('weapon',['assault_rifle','rifle'])
        root.assign_all_descendents()
        IDfineLabel2coarseLabel = {label : root.children_to_labels[root.closest_descendent_for(name).name] for label, name in enumerate(fine_names)}   

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        batch_names = [label2name[y.item()] for y in label]     
        batch_size = len(target)

        batch_start = time.time()   

        cross_entropy = 0
        cluster_cost = 0
        separation_cost = 0        
        l1 = 0
        noise_cross_ent = 0

        num_parents_in_batch = 0

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            
            _ = model(input) 

            for node in model.module.root.nodes_with_children():
                # get names specific to children
                children_idx = torch.tensor([name in node.descendents for name in batch_names])
                batch_names_coarsest = [node.closest_descendent_for(name).name for name in batch_names if name in node.descendents] # size of sum(children_idx)
                node_y = torch.tensor([node.children_to_labels[name] for name in batch_names_coarsest]).cuda() # size of sum(children_idx)
                node_logits = node.logits[children_idx]

                if len(node_y) > 0:
                    num_parents_in_batch += 1

                if node.name == "root":                    
                    root_y = node_y
                    #batch_names_coarse = [root.closest_descendent_for(name).name for name in batch_names if name in root.descendents] # size of sum(children_idx)
                    #root_y = torch.tensor([root.children_to_labels[name] for name in batch_names_coarse]).cuda() # size of sum(children_idx)                    

                if warm_up:                
                    if node.name == "root":
                        cross_entropy += torch.nn.functional.cross_entropy(node_logits,node_y) 
                        cluster_cost_, separation_cost_ = auxiliary_costs(node_y,node.num_prototypes_per_class,node.num_children(),node.prototype_shape,node.min_distances[children_idx,:])                                    
                        cluster_cost += cluster_cost_
                        separation_cost += separation_cost_  
                        #l1 += get_l1(model,node,mask=args.mask)
                        l1 += weight_reg(model,node)            

                if not warm_up:
                    cross_entropy += torch.nn.functional.cross_entropy(node_logits,node_y) * len(node_y) / dataloader.batch_size if len(node_y) > 0 else 0                    
                                
                    cluster_cost_, separation_cost_ = auxiliary_costs(node_y,node.num_prototypes_per_class,node.num_children(),node.prototype_shape,node.min_distances[children_idx,:])                                    
                    cluster_cost += cluster_cost_
                    separation_cost += separation_cost_               
                    #l1 += get_l1(model,node,mask=args.mask)            
                    l1 += weight_reg(model,node)

            
            preds_root, preds_joint = model.module.get_joint_distribution()


            # target_one_hot = torch.zeros((preds_joint.size(0),preds_joint.size(1))).cuda()
            # make_one_hot(target,target_one_hot)            
            # cross_entropy += CE(preds_joint,target_one_hot)
            
            # evaluation statistics
            _, coarse_predicted = torch.max(preds_root.data, 1)
            _, fine_predicted = torch.max(preds_joint.data, 1)
            
            if flat_model:
                coarse_probs = torch.zeros(batch_size,num_coarse)
                for j in range(len(label)):
                    # if fine_names[label[j].item()] == "revolver":
                    #     print([np.round(prob.item(),2) for prob in preds_joint[j,:]])
                    #     print(fine_names[torch.argmax(preds_joint[j,:])])                    
                    for k in range(num_fine):
                        coarse_id = IDfineLabel2coarseLabel[k]
                        coarse_probs[j,coarse_id] += preds_joint[j,k].item()
                coarse_predicted = torch.argmax(coarse_probs,1).cuda()            
            

            n_examples += target.size(0)
            coarse_correct = coarse_predicted == root_y
            fine_correct = fine_predicted == target
            n_coarse_correct += coarse_correct.sum().item()
            n_fine_correct += fine_correct.sum().item()    

            
            if class_acc:
                for j in range(len(target)):
                    coarse_class_correct[root_y[j]] += (1 if coarse_correct[j] else 0)
                    coarse_class_total[root_y[j]] += 1
                    fine_class_correct[target[j]] += (1 if fine_correct[j] else 0)
                    fine_class_total[target[j]] += 1
            #if is_train: print("batch accuracy: ", ((fine_predicted == target).sum().item()) / batch_size)


        if is_train:            
            loss = (20 * cross_entropy + args.lambda_cluster * cluster_cost + args.lambda_sep * separation_cost + 1e-2 * l1) / batch_mult
            #loss = (20 * cross_entropy + args.lambda_cluster * cluster_cost + args.lambda_sep * separation_cost + 5e-3 * l1) / batch_mult
            loss.backward()
            
            if CEDA:              
                noise = torch.stack([normalize(torch.rand((3,32,32)),mean,std) for n in range(batch_size)]).cuda()
                _ = model(noise) 
                for node in model.module.root.nodes_with_children():
                    noise_cross_ent += 1 * CE(node.logits,node.unif) # 1/10
                noise_cross_ent.backward()

            
            # optimizer.step()
            if (i+1) % batch_mult == 0:
                optimizer.step()
                optimizer.zero_grad()


        n_batches += 1        

        total_cross_entropy += cross_entropy.item()
        total_cluster_cost += cluster_cost.item() / num_parents_in_batch
        total_separation_cost += separation_cost.item() / num_parents_in_batch
        total_l1 += l1.item()
        total_noise_cross_ent += noise_cross_ent.item() if CEDA else 0

            
        del input
        del target, root_y, node_y
        del preds_root, preds_joint                
        del coarse_predicted
        del fine_predicted

        batch_end = time.time()
        #print('\tbatch time %.2f' % (batch_end-batch_start))
        

    end = time.time()

    log('\ttime: \t{0:.2f}'.format(end -  start))

    log('\tcross ent: \t{0:.2f}'.format(total_cross_entropy / n_batches))
    log('\tnoise cross ent: \t{0:.2f}'.format(total_noise_cross_ent / n_batches))
    log('\tcluster: \t{0:.2f}'.format(total_cluster_cost / n_batches))
    log('\tseparation: \t{0:.2f}'.format(total_separation_cost / n_batches))
    log('\tl1: \t{0:.2f}'.format(total_l1 / n_batches))        

    log('\tcoarse acc: \t{0:.2f}%'.format(n_coarse_correct / n_examples * 100))
    log('\tfine acc: \t{0:.2f}%'.format(n_fine_correct / n_examples * 100))

    if class_acc:
        log('\n')
        for c in range(len(coarse_names)):
            log('\tAccuracy of %5s : %2d %%' % (coarse_names[c], 100 * coarse_class_correct[c] / coarse_class_total[c]))    
        log('\n')
        for c in range(len(fine_names)):
            log('\tAccuracy of %5s : %2d %%' % (fine_names[c], 100 * fine_class_correct[c] / fine_class_total[c]))            

    #log('\tproto cluster: \t{0:.2f}'.format(proto_cluster_cost.item()))
    #p = model.module.prototype_vectors.view(model.module.num_prototypes, -1)
    #log('\tp dist pair: \t{0}'.format(torch.mean(list_of_distances(p, p)).item()))

    return n_fine_correct / n_examples, n_coarse_correct / n_examples    







def _OOD_test(model, dataloader, label2name, IDroot, OODroot, optimizer=None, class_specific=False, log=print, warm_up = False, CEDA = False, batch_mult = 1, class_acc = True):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    if not is_train: model.eval()
    start = time.time()
    n_examples = 0
    n_coarse_correct = 0
    n_fine_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_noise_cross_ent = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_l1 = 0
        
    coarse_names = OODroot.children_names()
    num_coarse = len(coarse_names)
    coarse_class_correct = np.array([0 for i in range(OODroot.num_children())])
    coarse_class_total = np.array([0 for i in range(OODroot.num_children())])

    fine_names = [x for x in label2name.values()]
    
    fine_class_correct = np.array([0 for i in range(len(fine_names))])
    fine_class_total = np.array([0 for i in range(len(fine_names))])


    fineLabel2coarseLabel = {label : OODroot.children_to_labels[OODroot.closest_descendent_for(name).name] for label, name in enumerate(fine_names)}
    IDfine_names = [node.name for node in model.module.root.nodes_without_children()]
    IDfine_names.sort()
    num_fine = (len(IDfine_names))

    IDfineLabel2coarseLabel = {label : IDroot.children_to_labels[IDroot.closest_descendent_for(name).name] for label, name in enumerate(IDfine_names)}    

    model.module.root.assign_unif_distributions()

    flat_model = not model.module.root.children[0].has_logits()    
    if flat_model: print("using flat model")

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        batch_names = [label2name[y.item()] for y in label]     
        batch_size = len(target)

        batch_start = time.time()   

        cross_entropy = 0
        cluster_cost = 0
        separation_cost = 0        

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:


            _ = model(input)                       
            root_y = torch.tensor([fineLabel2coarseLabel[y.item()] for y in label]).cuda()
            
            preds_root, preds_joint = model.module.get_joint_distribution()
            
                  
           
            # stats only for root
            cross_entropy += torch.nn.functional.cross_entropy(model.module.root.logits,root_y) 
            cluster_cost_, separation_cost_ = auxiliary_costs(root_y,model.module.root.num_prototypes_per_class,model.module.root.num_children(),
                model.module.root.prototype_shape,model.module.root.min_distances)
            cluster_cost += cluster_cost_
            separation_cost += separation_cost_        
                                
            # evaluation statistics

            if not flat_model:
                coarse_predicted = torch.argmax(preds_root.data, 1)
            else:
                coarse_probs = torch.zeros(batch_size,num_coarse)
                for j in range(len(label)):
                    # if fine_names[label[j].item()] == "revolver":
                    #     print([np.round(prob.item(),2) for prob in preds_joint[j,:]])
                    #     print(fine_names[torch.argmax(preds_joint[j,:])])                    
                    for k in range(num_fine):
                        coarse_id = IDfineLabel2coarseLabel[k]
                        coarse_probs[j,coarse_id] += preds_joint[j,k].item()
                coarse_predicted = torch.argmax(coarse_probs,1).cuda()

            coarse_correct = coarse_predicted == root_y                

            # this checks if the predicted fine class is a member of the coarse class
            # coarse_correct = torch.zeros(batch_size)
            # fine_predicted = torch.argmax(preds_joint.data, 1)
            # for j in range(len(target)):
            #     fine_pred = fine_predicted[j]
            #     coarse_target = root_y[j]
            #     if IDfine_names[fine_pred] in IDroot.children[coarse_target].descendents:
            #         # print("found that %s in %s" % (fine_names[fine_pred],OODroot.children[coarse_target].name))
            #         coarse_correct[j] += 1   
                
            
            

            n_examples += target.size(0)
            n_coarse_correct += coarse_correct.sum().item()


            if class_acc:
                for j in range(len(target)):
                    coarse_class_correct[root_y[j]] += (1 if coarse_correct[j] else 0)
                    coarse_class_total[root_y[j]] += 1
                    fine_class_correct[target[j]] += (1 if coarse_correct[j] else 0)
                    fine_class_total[target[j]] += 1
            #if is_train: print("batch accuracy: ", ((fine_predicted == target).sum().item()) / batch_size)


        n_batches += 1

        total_cross_entropy += cross_entropy.item()
        total_cluster_cost += cluster_cost.item()
        total_separation_cost += separation_cost.item()        

            
        del input
        del target, root_y
        del preds_root, preds_joint                
        del coarse_correct#,coarse_predicted

        batch_end = time.time()
        #print('\tbatch time %.2f' % (batch_end-batch_start))
        

    end = time.time()

    log('\ttime: \t{0:.2f}'.format(end -  start))

    log('\tcross ent: \t{0:.2f}'.format(total_cross_entropy / n_batches))
    log('\tnoise cross ent: \t{0:.2f}'.format(total_noise_cross_ent / n_batches))
    log('\tcluster: \t{0:.2f}'.format(total_cluster_cost / n_batches))
    log('\tseparation: \t{0:.2f}'.format(total_separation_cost / n_batches))      

    log('\tcoarse acc: \t{0:.2f}%'.format(n_coarse_correct / n_examples * 100))

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

    #log('\tproto cluster: \t{0:.2f}'.format(proto_cluster_cost.item()))
    #p = model.module.prototype_vectors.view(model.module.num_prototypes, -1)
    #log('\tp dist pair: \t{0}'.format(torch.mean(list_of_distances(p, p)).item()))

    return n_coarse_correct / n_examples   






def train(model, dataloader, label2name, optimizer, args, class_specific=False, log=print, warm_up = False):
    assert(optimizer is not None)
    log('train')
    return _train_or_test(model=model, dataloader=dataloader, label2name=label2name, optimizer=optimizer, args=args,
                          class_specific=class_specific, log=log, warm_up = warm_up, CEDA=args.CEDA, batch_mult = args.batch_mult)

def valid(model, dataloader, label2name, args, class_specific=False, log=print):
    log('valid')
    return _train_or_test(model=model, dataloader=dataloader, label2name=label2name, optimizer=None, args=args,
                          class_specific=class_specific, log=log)

def test(model, dataloader, label2name, args, class_specific=False, log=print, class_acc = False):
    #log('test')
    return _train_or_test(model=model, dataloader=dataloader, label2name=label2name, optimizer=None, args = args,
                      class_specific=class_specific, log=log, class_acc=class_acc)

def OOD_test(model, dataloader, label2name, args, IDroot, OODroot, class_specific=False, log=print, class_acc = True):
    #log('test')
    return _OOD_test(model=model, dataloader=dataloader, label2name=label2name,IDroot=IDroot, OODroot = OODroot, optimizer=None, 
                  class_specific=class_specific, log=log, class_acc=class_acc)





def auxiliary_costs(label,num_prototypes_per_class,num_classes,prototype_shape,min_distances):

    if label.size(0) == 0: 
        return 0, 0
    
    max_dist = prototype_shape[1]*prototype_shape[2]*prototype_shape[3]

    target = label.cuda()
    target_one_hot = torch.zeros(target.size(0), num_classes)
    target_one_hot = target_one_hot.cuda()                
    make_one_hot(target, target_one_hot)
    one_hot_repeat = target_one_hot.unsqueeze(2).repeat(1,1,num_prototypes_per_class).\
                        view(target_one_hot.size(0),-1)
    inverted_distances, _ = torch.max((max_dist - min_distances) * one_hot_repeat, dim=1)
    cluster_cost = torch.mean(max_dist - inverted_distances)

    flipped_one_hot_repeat = 1 - one_hot_repeat
    inverted_distances_to_nontarget_prototypes, _ = \
        torch.max((max_dist - min_distances) * flipped_one_hot_repeat, dim=1)
    separation_cost = torch.mean(inverted_distances_to_nontarget_prototypes)

    return cluster_cost, separation_cost
    

def get_l1(model,node, mask = True):

    if mask:
        identity = torch.eye(node.num_children()).cuda()
        repeated_identity = identity.unsqueeze(2).repeat(1,1,node.num_prototypes_per_class).\
                                view(node.num_children(), -1)
        l1_mask = 1 - repeated_identity
        weight = getattr(getattr(model.module,node.name + "_layer"),'weight') * l1_mask

    else:
        weight = getattr(getattr(model.module,node.name + "_layer"),'weight')
    
    return weight.norm(p=1)    


def weight_reg(model,node):
    # l1 reg on off-prototype weights, l2 on on-proto weights

    identity = torch.eye(node.num_children()).cuda()
    repeated_identity = identity.unsqueeze(2).repeat(1,1,node.num_prototypes_per_class).\
                            view(node.num_children(), -1)
    l1_mask = 1 - repeated_identity
    weight = getattr(getattr(model.module,node.name + "_layer"),'weight')
    off_weight = weight * l1_mask
    on_weight = weight * repeated_identity
    
    return off_weight.norm(p=1) + 1/10 * on_weight.norm(p=2)           

# warm only opts

def coarse_warm(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.root_prototype_vectors.requires_grad = True    
    for node in model.module.root.nodes_with_children():
        if node.name != "root":
            vecs = getattr(model.module,node.name + "_prototype_vectors")
            vecs.requires_grad = False
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = False                     
    log('coarse warm')

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True    
    for node in model.module.root.nodes_with_children():
        vecs = getattr(model.module,node.name + "_prototype_vectors")
        vecs.requires_grad = True
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = False                     
    log('warm')
   
# up to protos opts

def coarse_up_to_protos(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.root_prototype_vectors.requires_grad = True    
    for node in model.module.root.nodes_with_children():
        if node.name != "root":
            vecs = getattr(model.module,node.name + "_prototype_vectors")
            vecs.requires_grad = False
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = False                     
    log('coarse up to protos')

def up_to_protos(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    for node in model.module.root.nodes_with_children():
        vecs = getattr(model.module,node.name + "_prototype_vectors")
        vecs.requires_grad = True
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = False                     
    log('through protos')


# joint opts

def coarse_joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.root_prototype_vectors.requires_grad = True    
    for p in model.module.root_layer.parameters():
        p.requires_grad = True  
    for node in model.module.root.nodes_with_children():
        if node.name != "root":
            vecs = getattr(model.module,node.name + "_prototype_vectors")
            vecs.requires_grad = False
            layer = getattr(model.module,node.name + "_layer")
            for p in layer.parameters():
                p.requires_grad = False                     
    log('coarse joint')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    for node in model.module.root.nodes_with_children():
        vecs = getattr(model.module,node.name + "_prototype_vectors")
        vecs.requires_grad = True
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = True                     
    log('joint')


# last layer opts

def last_layers(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    for node in model.module.root.nodes_with_children():
        vecs = getattr(model.module,node.name + "_prototype_vectors")
        vecs.requires_grad = False
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = True                  
    log('last layers')



