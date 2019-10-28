import numpy as np

# imagenet
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
img_size = 224


#cifar10
# mean = (0.4914, 0.4822, 0.4465)
# std = (0.2023, 0.1994, 0.2010)
# img_size = 32

def preprocess(x, mean, std):
    for i in range(3):
        x[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return x

def undo_preprocess(x, mean, std):
    for i in range(3):
        x[:, i, :, :] = x[:, i, :, :] * std[i] +  mean[i]
        sub_zero = np.where(x[:, i, :, :] < 0)
        sup_one = np.where(x[:, i, :, :] > 1)
        x[:, i, :, :][sub_zero] = 0.
        x[:, i, :, :][sup_one] = 1.
    return x

def preprocess_input_function(x):
    return preprocess(x, mean=mean, std=std)

def undo_preprocess_input_function(x):
    return undo_preprocess(x, mean=mean, std=std)
