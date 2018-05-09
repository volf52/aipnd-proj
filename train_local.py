import os
import argparse
import time
import numpy as np

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

def main():
  args = get_arguments()
  dataloaders, dataset_sizes, data_transforms, image_datasets = get_data(args.data_directory)
  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
  save_dir = os.path.join(args.save_dir, 'checkpoint.pth')

  if os.path.exists(save_dir):
      model = load_checkpoint(save_dir, args.arch)
      classifier = model.classifier
  else:
    if args.arch == "vgg19":
        model = models.vgg19(pretrained = True)
        in_size = 25088
        classifier = nn.Sequential(OrderedDict([
                                ('0', nn.Linear(25088, int(args.hidden_units))),
                                ('1', nn.ReLU()),
                                ('2', nn.Dropout(0.5)),
                                ('3', nn.Linear(int(args.hidden_units), 102)),
                                ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        model = models.densenet121(pretrained=True)
        in_size = 1024
        classifier = nn.Sequential(OrderedDict([
            ('0', nn.Linear(1024, int(args.hidden_units))),
            ('1', nn.ReLU()),
            ('2', nn.Dropout(0.5)),
            ('3', nn.Linear(int(args.hidden_units), 102)),
            ('output', nn.LogSoftmax(dim=1))]))

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = classifier


  
  if args.gpu_av:
    model = model.cuda()
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.classifier.parameters(), lr=float(args.lr), momentum=0.9)

  model = train_model(model, criterion, optimizer, dataloaders, 
                      dataset_sizes, num_epochs=int(args.epochs), gpu_av=args.gpu_av)
  
  test_model(model, dataloaders['test'], dataset_sizes['test'], criterion, args.gpu_av)
  checkpoint = {'classifier_input_size': in_size,
              'output_size': 102,
              'optimizer_state' : optimizer.state_dict(),
              'classifier' : classifier,
              'state_dict': model.state_dict(),
             'data_transforms' : data_transforms['test'],
             'class_to_idx' : image_datasets['test'].class_to_idx,
             'arch' : args.arch
             }

  torch.save(checkpoint, save_dir)
  
  

def get_arguments():
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--save_dir", action="store", dest="save_dir", default="." , help = "Set directory to save checkpoints")
  parser.add_argument("--arch", action="store", dest="arch", default="vgg19" , help = "Set architechture('vgg19' or 'densenet121')")
  parser.add_argument("--learning_rate", action="store", dest="lr", default=0.01 , help = "Set learning rate")
  parser.add_argument("--hidden_units", action="store", dest="hidden_units", default=512 , help = "Set number of hidden units")
  parser.add_argument("--epochs", action="store", dest="epochs", default=5 , help = "Set number of epochs")
  parser.add_argument("--gpu", action="store_true", dest="gpu_av", default=False , help = "Wanna use GPU?")
  parser.add_argument('data_directory', action="store")
  
  return parser.parse_args()


def get_data(data_dir):
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'

  data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  }
  
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform = data_transforms[x]) for x in ['train', 'valid', 'test']}
  dataloaders    = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True) for x in ['train', 'valid', 'test']}
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

  return dataloaders, dataset_sizes, data_transforms, image_datasets

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=5, gpu_av=False):
  change_to = torch.FloatTensor
  if gpu_av:
    change_to = torch.cuda.FloatTensor
  for x in model.features:
    if type(x) == nn.modules.conv.Conv2d:
      x.weight.data = x.weight.data.type(change_to)
      if x.bias is not None:
        x.bias.data = x.bias.data.type(change_to)

  for x in model.classifier:
    if type(x) == nn.modules.linear.Linear:
      x.weight.data = x.weight.data.type(change_to)
      if x.bias is not None:
        x.bias.data = x.bias.data.type(change_to) 
          
  start = time.time()
  for epoch in range(num_epochs):
    print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    for phase in ['train', 'valid']:
        if phase == 'train':
          model = model.train()  # Set model to training mode
        else:
          model = model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in iter(dataloaders[phase]):
          if gpu_av:
            inputs = inputs.cuda()
            labels = labels.cuda()

          optimizer.zero_grad()

          # forward
          # track history if only in train
          if phase == 'train':
              curr = time.time() - start
              #print('\n {:.0f}m {:.0f}s'.format(curr // 60, curr % 60))
              inputs, labels = Variable(inputs), Variable(labels)
              outputs = model.forward(inputs)
              ps = torch.exp(outputs).data
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
          elif phase == 'valid':
              inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
              with torch.no_grad():
                outputs = model.forward(inputs)
                ps = torch.exp(outputs).data
                loss = criterion(outputs, labels)

          # statistics
          equals = (labels.data == ps.max(1)[1])
          running_loss += loss * inputs.size(0)
          running_corrects += equals.type_as(torch.FloatTensor()).sum()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]     
        epoch_loss = epoch_loss.data.cpu().numpy()         

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))


  time_elapsed = time.time() - start
  print('\nTraining complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))

  return model

def test_model(model, test_data, data_size, criterion, gpu_av):
    model = model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in iter(test_data):
        if gpu_av:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        with torch.no_grad():
            outputs = model.forward(inputs)
            ps = torch.exp(outputs).data
            equals = (labels.data == ps.max(1)[1])
            loss = criterion(outputs, labels)
        running_loss += loss * inputs.size(0)
        running_corrects += equals.type_as(torch.FloatTensor()).sum()

    test_loss = running_loss / data_size
    test_loss = test_loss.data.cpu().numpy()
    test_acc = running_corrects / data_size

    print('\n\n{} Loss: {:.4f} Acc: {:.4f}'.format('Test', test_loss, test_acc))

def load_checkpoint(filepath, arch):
    checkpoint = torch.load(filepath)
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    for x in model.features:
        if type(x) == nn.modules.conv.Conv2d:
            x.weight.data = x.weight.data.type(torch.DoubleTensor)
            if x.bias is not None:
                x.bias.data = x.bias.data.type(torch.DoubleTensor)

    for x in model.classifier:
        if type(x) == nn.modules.linear.Linear:
            x.weight.data = x.weight.data.type(torch.DoubleTensor)
            if x.bias is not None:
                x.bias.data = x.bias.data.type(torch.DoubleTensor) 

    
    return model


main()