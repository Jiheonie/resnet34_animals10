<p align="center">
 <h1 align="center">ResNet-34</h1>
</p>


## Introduction
Here is my python source code for ResNet-34 model - the model that won the the first prize ILSVRC and COCO 2015

## Dataset
The dataset used for training my model could be found at [Animals-10] https://www.kaggle.com/datasets/alessiocorrado99/animals10 or could be download by **python download_dataset.py**

## Categories
The table below shows 10 categories my model used:

|           |           | 
|-----------|:-----------:|
|    dog    |    cat    |  
|   horse   |    cow    |  
|  elephant |   sheep   |  
| butterfly |   spider  | 
|  chicken  |  squirrel | 

## Trained models
You could find my trained model at **trained_models/best_resnet34.pt**

## Training
After the raw dataset has been downloaded, you need to split dataset into training/test sets with ration 8:2 by simply run **dataset.py**.

## Experiments
The loss on training set and accuracy on test set curves for the experiment are shown below:

<img src="demo/train_loss.png" width="800"> 
<img src="demo/val_acc.png" width="800"> 

The confusion matrix of validate set are shown below:
<img src="demo/confusion_matrix.png" width="800"> 


## Requirements
* **python 3.11**
* **cv2 4.11**
* **pytorch 2.0** 
* **numpy**