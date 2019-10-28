# Interpretable Image Recognition with Hierarchical Prototypes
This repository contains contain code for [Interpretable Image Recognition with Hierarchical Prototypes](https://arxiv.org/abs/1906.10651)

## Repository Structure

```
|__ HPnet/ --> Directory with HPnet code
        |__ saved_models_8protos1/ --> Directory with pre-trained HPnet model and saved prototype images & neighbors
        |__ *.ipynb --> iPython notebooks for interpeting model prototypes & classifications, fitting and evaluating novel class detectors
        |__*.py --> HPnet framework code
|__ vgg/ --> Directory with code for training vgg base model and evaluating accuracy according to hierarchical class organization in HPnet
```

## Requirements

- Python 3.6
- PyTorch 1.3
- torchvision 0.4.1
- SciPy 1.0.0
 
## Data

We perform experiments on a subset of ImageNet 2012 classes, described in detail in our paper. We treat 15 classes as in-distribution, and 15 as "novel" classes.

Our code assumes the directory structure

```
|__ datasets/ --> Directory with HPnet code
        |__ Imagenet/ --> 
                |train --> 1250 train images per class
                |valid --> 50 validation images per class
                |test --> 50 test images per class
                |OOD/ --> Folder for novel class data
                      |train --> 1250 train images per class
                      |valid --> 50 validation images per class
                      |test --> 50 test images per class
                |OODall/ --> Folder for novel class accuracy evaluation (lacks train-val-test slit of OOD)
                 
```

## Reproducing Experiments 

To train a new HPnet model using our class hierarchy, and assuming you have the train/val/test data collected and organized as above, run

```
cd HPnet
python main.py 
```

To view and interpret model prototypes, which can be done without training a model, using our provided images), use ```view_protos.ipynb``` and ```nearest_neighbors.ipynb```. 

For training and evaluate novel class detectors, see ```novel_class_detection.ipynb```

For a case study of the model in action, see ```case_study.ipynb```. 

