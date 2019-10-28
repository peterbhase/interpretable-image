import numpy as np
import torch 

class Node:

    def __init__(self, name, parent = None, label = None):

        self.parent = parent
        self.children = []
        self.children_to_labels = {}
        self.name = name
        self.label = label


    def add_children(self, names, labels = None):
        if type(names) is not list:
            names = [names];
        if labels is None:
            labels = [i for i in range(len(self.children),len(self.children)+len(names))]
        names.sort()
        for i in range(len(names)):
            self.children.append( Node(names[i], parent=self, label = labels[i]))    
            self.children_to_labels.update({names[i] : labels[i]})

    def get_child(self, name):
        for child in self.children:
            if child.name == name:
                return child    

    def children_names(self):
        return([child.name for child in self.children])

    # def assign_children_names(self):
    #     self.children_names = self.children_names()

    # def assign_all_children_names(self):        
    #     active_nodes = []
    #     active_nodes += [self]
    #     while len(active_nodes) > 0:
    #         for node in active_nodes:
    #             node.assign_children_names()
    #             nodel.children_names.sort()
    #         new_active_nodes = [] 
    #         for node in active_nodes:
    #             new_active_nodes += node.children
    #         active_nodes = new_active_nodes           

    def get_node(self,name):                
        active_nodes = [self]
        while True:
            for node in active_nodes:
                if node.name == name:
                    return node
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes
            if len(active_nodes) == 0:
                print("node for " + name + " not found")
                break

    def get_node_attr(self,name,attr):
        node = self.get_node(name)
        return getattr(node,attr)

    def set_node_attr(self,name,attr,value):
        node = self.get_node(name)
        return setattr(node,attr,value)        
                
    def num_children(self):
        return(len(self.children))

    def class_to_num_children(self):
        class_to_num = {}
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                class_to_num.update({node.name : node.num_children()})
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return class_to_num

    def class_to_proto_shape(self, x_per_child = 1, dimension = 512):
        class_to_shape = {}
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                if node.num_children() > 0:
                    class_to_shape.update({node.name : (x_per_child * node.num_children(),dimension,1,1)})
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return class_to_shape        

    def classes_with_children(self):
        classes = []
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                if node.num_children() > 0:
                    classes.append(node.name)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return classes

    def nodes_with_children(self):        
        nodes = []
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                if node.num_children() > 0:# and node.name != "root":
                    nodes.append(node)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return nodes        

    def add_children_to(self,name,children):
        node = self.get_node(name)
        node.add_children(children)

    def parents(self):
        parents = []
        parent = node.parent
        parents += parent
        while parent.parent is not None:
            parent = parent.parent
            parents += parent
        return parents


    def assign_descendents(self):
        active_nodes = []
        active_nodes += self.children
        descendents = set()
        while len(active_nodes) > 0:
            for node in active_nodes:
                descendents.add(node.name)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        self.descendents = descendents

    def assign_all_descendents(self):
        active_nodes = []
        active_nodes += [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                node.assign_descendents()
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                            


    def is_descendent(name):
        return name in self.descendents        


    def closest_descendent_for(self,name):
    	if name in self.children_names(): 
    		return self.get_node(name)
    	else:
        	return [child for child in self.children if name in child.descendents][0]


    def has_logits(self):
        return self.num_children() > 1

    def get_distribution(self):                
        if self.has_logits():
            return torch.nn.functional.softmax(self.logits,1)
        else:
            batch_size = self.logits.size(0)
            return torch.ones((batch_size,1))

        
    def distribution_over_furthest_descendents(self,batch_size):
        if not self.has_logits():
            return torch.ones(batch_size,1).cuda()
        else:
            return torch.cat([torch.nn.functional.softmax(self.logits,1)[:,i].view(batch_size,1) * self.children[i].distribution_over_furthest_descendents(batch_size) for i in range(self.num_children())],1)            

    def names_of_joint_distribution(self):
        if self.num_children() == 1:
            return [self.children[0].name]
        elif self.num_children() == 0:
            return [self.name]
        else:
            return [child.names_of_joint_distribution() for child in self.children]


    def unwrap_names_of_joint(self,names):
        # this is the worst thing i've ever written
        new_list = []
        for item in names:
            if type(item) is not list:
                new_list.append(item)
            else:
                for subitem in item:
                    if type(subitem) is not list:
                        new_list.append(subitem)
                    else:
                        for subsubitem in subitem:
                            if type(subsubitem) is not list:
                                new_list.append(subsubitem)
                            else:
                                for subsubsubitem in subsubitem:
                                    if type(subsubsubitem) is not list:
                                        new_list.append(subsubsubitem)
        return new_list


    def assign_unif_distributions(self):
        for node in self.nodes_with_children():
            node.unif = (torch.ones(node.num_children()) / node.num_children()).cuda()


    def nodes_without_children(self):        
        nodes = []
        active_nodes = [self]
        while len(active_nodes) > 0:
            for node in active_nodes:
                if not node.has_logits():
                    nodes.append(node)
            new_active_nodes = [] 
            for node in active_nodes:
                new_active_nodes += node.children
            active_nodes = new_active_nodes                    
        return nodes 


    def __str__(self):
        return "Node for " + self.name



if __name__ == "__main__":
    x = Node('cats',0)
    x.add_children(['leopard','tiger'])
    x.add_children(['lion','house cat'])
    lion = x.get_child('lion')
    lion.add_children('test')
    x.add_children_to('house cat',['tony','strappy'])
    x.add_children_to('tony',['paws','tail'])
    x.add_children_to('paws',['terrible'])
    #print(x.children[2].children[0].name)
    # print(x.children_names())
    # print(lion.children_names())
    # print(lion.parent.name)
    # print(x.get_node('test'))
    # setattr(x,"test",2)
    # print(x.test)
    # print(x.name)
    # print(x.children_to_labels)
    # print(x.children)
    print(x.get_child("lion").name)
    print(getattr(x,"children"))
    # print(x.class_to_num_children()) 

    full_list = x.names_of_joint_distribution()
    print(full_list)
    print(x.unwrap_names_of_joint(full_list))
    #print([item for sublist in full_list for item in sublist])
    
    # root = Node('root')
    # root.add_children(['vehicle','animal'])
    # root.add_children_to('vehicle',['airplane','automobile','ship','truck'])
    # root.add_children_to('animal',['bird','cat','deer','dog','frog','horse'])
    


