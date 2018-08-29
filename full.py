# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import json
import time
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import Image
import numpy as np
import os
%cd "C:\\Users\\tarek\\workspace\\conv-assignment\\aipnd-project"
import helper

# %%
data_dir = 'C:/Users/tarek/workspace/conv-assignment/aipnd-project/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(45),
                                      transforms.RandomHorizontalFlip(p=0.4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                              (0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225))])

data_transforms_test = transforms.Compose([transforms.Resize(224),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                                   (0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225))])

# TODO: Load the datasets with ImageFolder
image_datasets_train = datasets.ImageFolder(train_dir,
                                            transform=data_transforms)
image_datasets_valid = datasets.ImageFolder(valid_dir,
                                            transform=data_transforms)
image_datasets_test = datasets.ImageFolder(test_dir,
                                           transform=data_transforms_test)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = DataLoader(image_datasets_train, batch_size=64, shuffle=True)
validloader = DataLoader(image_datasets_valid, batch_size=64, shuffle=True)
testloader = DataLoader(image_datasets_test, batch_size=32, shuffle=False)

data_iter = iter(testloader)
images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# %%
# TODO: Build and train your network
model = models.vgg19(pretrained=True)
model

# %%
# Freeze parameters in VGG so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1000)),
                          ('relu', nn.ReLU()),
                          ('d1', nn.Dro pout(p=0.3)),
                          ('fc2', nn.Linear(1000, len(cat_to_name))),
                          # ('d2', nn.Dropout(p=0.2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

# TODO: Do validation on the test set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


# %%
model.to(device)

epochs = 1
steps = 0
print_every = 40
for e in range(epochs):
    running_loss = 0
    tot_time = time.time()
    model.train()
    for inputs, labels in iter(trainloader):
        steps += 1
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        start = time.time()
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

            running_loss = 0

            # Make sure training is back on
            model.train()
            running_loss = 0
    print("Time per epoch = {:.3f}".format(time.time()-tot_time), "seconds.")


# %%
# TODO: Do validation on the test set

correct = 0
total = 0
model.to(device)
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the', len(testloader), 'test images: %d %%'
      % (100 * correct / total))
# %%

# TODO: Save the checkpoint
model.class_to_idx = image_datasets_train.class_to_idx

checkpoint = {
              'state_dict': model.state_dict(),
              'image_datasets': model.class_to_idx,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'classifier': model.classifier,
             }

torch.save(checkpoint, 'checkpoint.pth')
# %%
# Loading the checkpoint


def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        epochs = checkpoint['epochs']
        model = models.vgg19(pretrained=True)
# =============================================================================
#         classifier = nn.Sequential(OrderedDict([
#                           ('fc1', nn.Linear(25088, 10000)),
#                           ('relu', nn.ReLU()),
#                           ('d1', nn.Dropout(p=0.2)),
#                           ('fc2', nn.Linear(10000, 102)),
#                           ('d2', nn.Dropout(p=0.2)),
#                           ('output', nn.LogSoftmax(dim=1))
#                           ]))
# =============================================================================
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['image_datasets']
        # optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer = checkpoint['optimizer']
        for param in model.parameters():
            param.requires_grad = False
        return model


model = load_checkpoint('checkpoint.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'C:/Users/tarek/workspace/conv-assignment/aipnd-project/flowers'
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# %%

# =============================================================================
# Image Preprocessing
# You'll want to use PIL to load the image (documentation). It's best to write
# a function that preprocesses the image so it can be used as input for the
# model. This function should process the images in the same manner used for
# training.
#
# First, resize the images where the shortest side is 256 pixels, keeping the
# aspect ratio. This can be done with the thumbnail or resize methods.
# Then you'll need to crop out the center 224x224 portion of the image.
#
# Color channels of images are typically encoded as integers 0-255, but the
# model expected floats 0-1. You'll need to convert the values. It's easiest
# with a Numpy array, which you can get from a PIL image like so
# np_image = np.array(pil_image).
#
# As before, the network expects the images to be normalized in a specific way.
# For the means, it's [0.485, 0.456, 0.406] and for the standard deviations
# [0.229, 0.224, 0.225]. You'll want to subtract the means from each color
# channel, then divide by the standard deviation.
#
# And finally, PyTorch expects the color channel to be the first dimension but
# it's the third dimension in the PIL image and Numpy array. You can reorder
# dimensions using ndarray.transpose. The color channel needs to be first
# and retain the order of the other two dimensions.
# =============================================================================


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Resizing to 256 pixels while keeping aspect ratio
    min_length = 256
    img = Image.open(image)
    width, height = img.size    # Get dimensions
    if width < height:
        img.thumbnail((min_length, height))
    else:
        img.thumbnail((width, min_length))

    # Center crop
    if min(width, height) < 224:    # Skips cropping pictures with less than
                                    # 224 pixels in either dimension
        return
    w, h = 224, 224     # New dimensions for center crop
    width, height = img.size    # Get new dimensions after resizing
    left = (width - w) // 2
    top = (height - h) // 2
    right = (width + w) // 2
    bottom = (height + h) // 2
    img = img.crop((left, top, right, bottom))
    # normalizing values of color channels
    # converting to tensor to normalize
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    img = normalize(img)
    img = np.array(img)
    img_trans = img.transpose()
    return img_trans


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    # image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when
    # displayed
    image = np.clip(image, 0, 1)
    plt.title(title)
    ax.imshow(image)

    return ax


img = (data_dir + '/test' + '/1/' + '/image_06752.jpg')

# Test function on a single picture
img = process_image(img)
img.shape
imshow(img)

# Looping through all images in a folder

directory = data_dir + "/test/1"

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        imshow(process_image(os.path.join(directory, filename)))
    else:
        continue
# %%
# =============================================================================
# Class Prediction
# Once you can get images in the correct format, it's time to write a function
# for making predictions with your model. A common practice is to predict the
# top 5 or so (usually called top- KK ) most probable classes. You'll want to
# calculate the class probabilities then find the  KK  largest values.
#
# To get the top  KK  largest values in a tensor use x.topk(k). This method
# returns both the highest k probabilities and  the indices of those
# probabilities corresponding to the classes. You need to convert from these
# indices to the actual class labels using class_to_idx which hopefully you
# added to the model or from an ImageFolder you used to load the data
# (see here). Make sure to invert the dictionary so you get a mapping from
# index to class as well.
#
# Again, this method should take a path to an image and a model checkpoint,
# then return the probabilities and classes.
#
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# =============================================================================

idi = model.class_to_idx
inv = {v: k for k, v in idi.items()}    # inverting a dictionary as seen in
# https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
model.to(device)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep
        learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    # Test out your network!
    model.eval()

    img = process_image(image_path)
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    img = img.unsqueeze(0)  # Adds a singleton dimension to make up for
    # the batch size value from the training input
    img = img.to(device)
    with torch.no_grad():
        output = model.forward(img)

    out = torch.exp(output)
    out = out.topk(topk)
    probs, classes = out
    probs, classes = np.array(probs), np.array(classes)
    probs, classes = np.squeeze(probs, 0), np.squeeze(classes, 0)
    cl = []
    for k in classes:
        cl.append(inv[k])
    return probs, cl


img = (data_dir + '/test' + '/1/' + '/image_06752.jpg')
probs, classes = predict(img, model)
print(probs)
print(classes)

# %%
# =============================================================================
# Sanity Checking
# Now that you can use a trained model for predictions, check to make sure it
# makes sense. Even if the testing accuracy is high, it's always good to check
# that there aren't obvious bugs. Use matplotlib to plot the probabilities for
# the top 5 classes as a bar graph, along with the input image. It should look
# like this:
#
# You can convert from the class integer encoding to actual flower names with
# the cat_to_name.json file (should have been loaded earlier in the notebook).
# To show a PyTorch tensor as an image, use the imshow function defined above.
# =============================================================================


# TODO: Display an image along with the top 5 classes
# img_path = (data_dir + '/test/23/image_03409.jpg')
# img_path = ("C:/Users/tarek/Downloads/flower.jpg")
img_path = (data_dir + '/test/18/image_04254.jpg')
img = process_image(img_path)
img.shape
imshow(img, title=cat_to_name[img_path.split("/")[8]])
probs, classes = predict(img_path, model)
names = []
for k in classes:
    names.append(cat_to_name[k])

# Fixing random state for reproducibility
np.random.seed(42)


plt.rcdefaults()
fig, ax = plt.subplots()

y_pos = np.arange(len(names))


ax.barh(y_pos, probs, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Probabilities')
ax.set_title('How confident is this model, really?')
plt.xlim([0.0, 1.0])   # set the range of the x-axis between 0 & 1
plt.show()
