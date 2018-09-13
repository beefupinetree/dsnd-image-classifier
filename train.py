import argparse
import json
import time
import os

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("data_dir",
                    help="directory with the data",
                    type=str)
parser.add_argument("-s", "--save_dir",
                    help="directory where the model is saved",
                    type=str,
                    action='store')
parser.add_argument("-a", "--arch",
                    help="architecture of convolutional neural network",
                    type=str,
                    action='store',
                    default="vgg19")
parser.add_argument("-l", "--learning_rate",
                    help="learning rate of the optimizer",
                    type=float,
                    action='store',
                    default=0.001)
parser.add_argument("--hidden_units",
                    help="number of neurons per hidden layer",
                    type=int,
                    action='store',
                    default=1000)
parser.add_argument("-e", "--epochs",
                    help="number of training epochs",
                    type=int,
                    action='store',
                    default=3)
parser.add_argument("--gpu",
                    help="selects gpu for training, if available",
                    action="store_true")

args = parser.parse_args()


data_dir = args.data_dir
train_dir = os.path.join(data_dir,"train")
valid_dir = os.path.join(data_dir,"valid")
test_dir = os.path.join(data_dir,"test")

# %%
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

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# %%
# TODO: Build and train your network
model = getattr(models, args.arch)
model = model(pretrained=True)

# %%
# Freeze parameters in VGG so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('d1', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

# TODO: Do validation on the test set

device = torch.device("cuda:0" if torch.cuda.is_available() or args.gpu else
                      "cpu")

criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)


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


model.to(device)

epochs = args.epochs
steps = 0
print_every = 32
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

model.class_to_idx = image_datasets_train.class_to_idx

checkpoint = {
              'state_dict': model.state_dict(),
              'image_datasets': model.class_to_idx,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'classifier': model.classifier,
              'model': args.arch,
             }

torch.save(checkpoint, args.save_dir + '\\checkpoint.pth' if args.save_dir
           else 'checkpoint.pth')
