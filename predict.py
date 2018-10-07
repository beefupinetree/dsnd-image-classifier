import argparse
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("img_path",
                    help="path to the image",
                    type=str)
parser.add_argument("checkpoint",
                    help="name of the model checkpoint to load",
                    type=str,
                    action='store')
parser.add_argument("--top_k",
                    help="top k probable matches",
                    type=int,
                    action='store',
                    default=1)
parser.add_argument("-category_names",
                    help="mapping of categories to real names",
                    type=str)
parser.add_argument("--gpu",
                    help="selects gpu for predicting, if available",
                    action="store_true")

args = parser.parse_args()

# %%


def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = getattr(models, checkpoint['model'])
        model = model(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['image_datasets']
        for param in model.parameters():
            param.requires_grad = False
        return model


model = load_checkpoint(args.checkpoint + '.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() or args.gpu else
                      "cpu")
with open(args.category_names if args.category_names
          else 'cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# %%


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


# %%

idi = model.class_to_idx
inv = {v: k for k, v in idi.items()}    # inverting a dictionary as seen in
# https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
model.to(device)


def predict(image_path, model, topk):
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


# %%

# TODO: Display an image along with the top 5 classes

img = process_image(args.img_path)
probs, classes = predict(args.img_path, model, args.top_k)

for i in range(0, args.top_k):
    print("{}. the flower power of '{}' is {:.0f}%"
          .format(i + 1, cat_to_name[classes[i]], round(probs[i] * 100, 0)))
