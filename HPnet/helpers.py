import os
import torch

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target_copy = torch.LongTensor(len(target))
    target_copy.copy_(target)
    #target = target.copy().view(-1,1)
    target_copy = target_copy.view(-1,1).cuda()
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target_copy, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')


def adjust_learning_rate(optimizers):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr_ = lr * (0.1 ** (epoch // decay))    
    for optimizer in  optimizers:
        for param_group in optimizer.param_groups:
            new_lr = param_group['lr'] * .1
            param_group['lr'] = new_lr



