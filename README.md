# Image classifier

It can learn from any set of labeled images and be used to label new images. The hyper-parameters can also be modified by the user from the command line itself. Finally the main model itself can be chosen by the user from all the possibilities within the TorchVision.Models library.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The necessary python libraries are:

```python
torch
torchvision
numpy
```
### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

To install `Torch` and `Tochvision` using Anaconda:
```bash
conda install pytorch cuda92 -c pytorch
conda install torchvision -c soumith
```
Using pip:

```bash
pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip install torchvision
```
## Quickstart
The below terminal commands will install the necessary packages in Git Bash and train a convolutional network on Oxford university's [flower data](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) using default values.
```bash
cd ~
conda install pytorch cuda92 -c pytorch
pip install torchvision
git clone "https://github.com/beefupinetree/dsnd-image-classifier.git"
cd dsnd-image-classifier
wget "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
tar xzvf flower_data.tar.gz
rm flower_data.tar.gz
mkdir flowers
mv -t flowers test train valid
python train.py flowers
```

<p align="center"><img src="/img/train.gif?raw=true"/></p>

You can access your newly trained model to classify any JPG picture, including all the ones in the 'Test' folder.
```bash
python predict.py "$(pwd)/flowers/test/59/image_05020.jpg" checkpoint --top_k 5 --gpu
```
<p align="center"><img src="/img/predict.gif?raw=true"/></p>
## Images

You must have a collection of labeled images. For example, if we were classifying cats and dogs, each folder must contain instances of each kind of picture. The folder structure must stay as follows:
* cats_dogs
 * test
  	* 1
  	* 2
 * train
  	* 1
  	* 2
 * valid
  	* 1
	* 2

The folders labeled '1' only contain pictures of cats and folders labeled '2' only contain pictures of dogs.

Next we will need a JSON file assigning each number to the appropriate label. In this example it would be:
```python
{"1": "Cat", "2": "Dog"}
```
## Training
To train the model on your data, run the following command in the shell:

```bash
python train.py <command> [options]
```

The command has one mandatory argument:

```
data_dir	Directory with the labeled images within separate folders for 'test, train, valid'	[string]
```

The options for training are:

```
-s, --save_dir  	Directory where the trained model is saved	[string] [default: current directory]
-a, --arch  		Architecture of convolutional neural network	[string] [default: vgg19]
-l, --learning_rate 	Learning rate of the optimizer			[float]	 [default: 0.001]
	--hidden_units	Number of neurons per hidden layer		[int]	 [default: 1000]
-e, --epochs		Number of training epochs			[int]	 [default: 3]
	--gpu		Automatically selects gpu for training, if available
```

Examples

```bash
python train.py flowers -e 10 --gpu		Trains a model on the data in the flowers directory by using the GPU and training the model for 10 epochs.
python train.py flowers -a vgg13 --gpu	Trains a neural network on the GPU with a VGG13 architecture
```

The trained convolutional neural network architectures available in the Torchvision library can be found [here](https://pytorch.org/docs/0.3.0/torchvision/models.html). There are multiple versions of:

1. VGG
2. ResNet
3. SqueezeNet
4. Densenet

## Predicting

We can then use our brand new model to identify whatever it was trained on. Be it cats & dogs, articles or clothing, or types of flowers. To do that, run the following command in the shell:

```bash
python predict.py <command> [options]
```

The command has two mandatory arguments:

```
img_path	Path to the image			[string]
checkpoint	Name of the model checkpoint to load	[string]
```

The options for training are:
```
-category_names 	Mapping of categories to real names in JSON		[string]
--top_k  		Top 'k' probable matches				[int] [default: 1]
--gpu			Automatically selects gpu for predicting, if available
```

Examples

```bash
python predict.py animalpic.jpg checkpoint10 --top_k 2 --gpu	Uses the saved model in 'checkpoint10' to predict whether 'animalpic' is a picture of a cat or a dog
python predict.py animalpic.jpg checkpoint10 -category_names cat_to_name.json --gpu	Same thing as above, only this one outputs the names of the categories instead of their number. So we will see 'Cat' and 'Dog' instead of '1' and '2'
```
