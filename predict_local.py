import os
import argparse
import json
import numpy as np

from PIL import Image

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms

def main():
    args = get_arguments()
    model = load_checkpoint(args.checkpoint)
    model.idx_to_class = dict([[v,k] for k,v in model.class_to_idx.items()])

    if args.gpu_av:
        model = model.cuda()

    with open(args.cat_file, 'r') as f:
        cat_to_name = json.load(f)
    
    a, b = predict(args.input, model, args.gpu_av, topk=int(args.top_k))
    b = [model.idx_to_class[x] for x in b]
    print(a)
    print(b)
    print([cat_to_name.get(x, 'NotFound') for x in b])




def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--category_names ", action="store", dest="cat_file", default="cat_to_name.json" , help = "Categories to names")
    parser.add_argument("--top_k", action="store", dest="top_k", default=5 , help = "Set number of results to return")
    parser.add_argument("--gpu", action="store_true", dest="gpu_av", default=False , help = "Wanna use GPU?")
    parser.add_argument('input', action="store")
    parser.add_argument('checkpoint', action="store")

    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == "vgg19":
        model = models.vgg19(pretrained=True)
    elif checkpoint['arch'] == "densenet121":
        model = models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for x in model.features:
        if type(x) == nn.modules.conv.Conv2d:
            x.weight.data = x.weight.data.type(torch.FloatTensor)
            if x.bias is not None:
                x.bias.data = x.bias.data.type(torch.FloatTensor)

    for x in model.classifier:
        if type(x) == nn.modules.linear.Linear:
            x.weight.data = x.weight.data.type(torch.FloatTensor)
            if x.bias is not None:
                x.bias.data = x.bias.data.type(torch.FloatTensor) 
    
    
    return model


def process_image(img):
    tr1 = transforms.ToTensor()
    ratio = img.size[1] / img.size[0]
    new_x = 256
    new_y = int(ratio * new_x)
    img = img.resize((new_x, new_y))
    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    cropped = img.crop(
        (
        half_the_width - 112,
        half_the_height - 112,
        half_the_width + 112,
        half_the_height + 112
        )
                        )
    
    np_image = np.array(cropped)
    np_image = np.array(np_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np_image - mean) / std
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)

def predict(image_path, model, gpu_av, topk=5):
    img = None
    with Image.open(image_path) as im:
        img = process_image(im)
    
    img = img.type(torch.cuda.FloatTensor)
    with torch.no_grad():
        img = Variable(img.unsqueeze(0), requires_grad=False)
        if gpu_av:
            img = img.cuda()
        output = model.forward(img)
        ps = torch.exp(output)
    probs, indeces = ps.topk(topk)
    return probs.data[0].cpu().numpy(), indeces.data[0].cpu().numpy()

main()