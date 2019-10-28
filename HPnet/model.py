import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import time

from receptive_field import compute_proto_layer_rf_info

img_size = 224

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG_proto(nn.Module):

    def __init__(self, features, root, proto_layer_rf_info, num_prototypes_per_class, prototype_dimension, init_weights=True):
        super(VGG_proto, self).__init__()

        self.root = root
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototype_dimension = prototype_dimension    

        for name, num_children in root.class_to_num_children().items():
            setattr(self,"num_" + name, num_children)
                

        for name,shape in root.class_to_proto_shape(x_per_child=num_prototypes_per_class, dimension=prototype_dimension).items():
            setattr(self,name + "_prototype_shape",shape)
            setattr(self,"num_" + name + "_prototypes",shape[0])
            setattr(self,name + "_prototype_vectors", nn.Parameter(torch.rand(shape), requires_grad=True))
            setattr(self,name + "_layer", nn.Linear(shape[0], getattr(self,"num_" + name), bias = False))
            setattr(self,"ones_" + name, nn.Parameter(torch.ones(shape), requires_grad=False))
            
            root.set_node_attr(name,"num_prototypes_per_class",num_prototypes_per_class)
            root.set_node_attr(name,"prototype_shape",shape)        


        self.epsilon = 1e-4

        self.proto_layer_rf_info = proto_layer_rf_info

        self.features = features # this has to be named features to allow the precise loading
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=prototype_dimension, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=prototype_dimension, out_channels=prototype_dimension, kernel_size=1),
            nn.Sigmoid()
        )  

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        conv_features = self.conv_features(x)

        for node in self.root.nodes_with_children():
            self.classifier(conv_features,node)

        return "remember logits and distances attached to nodes"


    def classifier(self, conv_features, node):
        distances = self.prototype_distances(conv_features, node.name)
        min_distances = -nn.functional.max_pool2d(-distances,
                                                  kernel_size=(distances.size()[2], distances.size()[3])) # global max pooling
        min_distances = min_distances.view(-1, getattr(self,"num_" + node.name + "_prototypes"))
        prototype_activations = torch.log(1 + (1 / (min_distances + self.epsilon)))
        logits = getattr(self, node.name + "_layer")(prototype_activations)
        setattr(node,"logits",logits)
        setattr(node,"min_distances",min_distances)
        return None


    def _l2_convolution(self, x, name):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''

        x2 = x ** 2
        x2_patch_sum = nn.functional.conv2d(input=x2, weight= getattr(self,"ones_" + name))
        p2 = getattr(self,name + "_prototype_vectors") ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3)) # p2 is a vector of shape (num_prototypes,)
        xp = nn.functional.conv2d(input=x, weight= getattr(self,name + "_prototype_vectors"))
        p2_reshape = p2.view(-1, 1, 1)
        intermediate_result = - 2 * xp + p2_reshape # use broadcast
        distances = nn.functional.relu(x2_patch_sum + intermediate_result) # x2_patch_sum and intermediate_result are of the same shape       

        return distances

    def conv_features(self, x):
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    def prototype_distances(self, conv_features, name):
        '''
        x is the raw input
        '''
        x = self._l2_convolution(conv_features, name)
        return x


    def remaining_layers(self, x):
        x = -nn.functional.max_pool2d(-x, kernel_size=(x.size()[2], x.size()[3])) # global max pooling
        x = x.view(-1, self.num_prototypes)
        x = - torch.log(x + self.epsilon)
        x = self.last_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules(): # Returns an iterator over all modules in the network.   
            if len(m._modules) > 0: # skip anything that's not a single layer
                continue
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)            
            elif isinstance(m, nn.Linear):
                identity = torch.eye(m.out_features)
                repeated_identity = identity.unsqueeze(2).repeat(1,1,self.num_prototypes_per_class).\
                                            view(m.out_features, -1)
                m.weight.data.copy_(1.5 * repeated_identity - 0.5)

                
    def _smart_init_fine_protos(self):
        # init the fine protos around the coarse protos belonging to their respective classes        
        # for each coarse class, iterate through its coarse protos dropping one new fine proto near it each time
        
        sd = .03

        for node in self.root.nodes_with_children():
            if node.name != "root":
                prototype_vectors = getattr(self,node.name + "_prototype_vectors")
                parent_prototype_vectors = getattr(self,node.parent.name + "_prototype_vectors")
                for i in range(prototype_vectors.size(0)):
                    j = i % self.num_prototypes_per_class
                    mean = torch.rand((self.prototype_dimension,1,1))
                    mean.data.copy_(parent_prototype_vectors[j,:,:,:].data)
                    prototype_vectors[i,:,:,:].data.copy_(nn.init.normal_(mean,sd))


    def get_joint_distribution(self):
           

        batch_size = self.root.logits.size(0)

        #top_level = torch.nn.functional.softmax(self.root.logits,1)            
        top_level = self.root.logits
        bottom_level = self.root.distribution_over_furthest_descendents(batch_size)    

        names = self.root.unwrap_names_of_joint(self.root.names_of_joint_distribution())
        idx = np.argsort(names)

        bottom_level = bottom_level[:,idx]        
        
        return top_level, bottom_level
            
                    

def make_layers(cfg, batch_norm=False):
    # cfg configuration
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers) # unravel the list to arguments of Sequential

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '256': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    '128': [64, 64, 'M', 128, 128, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg16_proto(root, pretrained=False, num_prototypes_per_class = 1, prototype_dimension = 512, model_path = None, resume_path = None, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    cfg_ = cfg['D']        
    model = VGG_proto(features=make_layers(cfg_), root=root, num_prototypes_per_class = num_prototypes_per_class, prototype_dimension = prototype_dimension,
                      proto_layer_rf_info=compute_proto_layer_rf_info(img_size, cfg_, 1), init_weights=True)
    
    if pretrained:
        
        if resume_path is not None:
            print("loading model from %s" % resume_path)
            model.load_state_dict(torch.load(resume_path))
        
        else: # expects path to VGG model
            print("loading model from %s" % model_path)
            state_dict = torch.load(model_path)
            
            new_state_dict = dict()
            for k, v in state_dict.items():            
                name = k
                new_state_dict[name] = v
        
            keys = [key for key in new_state_dict.keys()]
            for key in keys:
                if key.startswith('classifier'): # the classifier weights are the fully connected layers in vgg which we have removed
                    del new_state_dict[key]
        
            model.load_state_dict(new_state_dict, strict = False)

    return model

