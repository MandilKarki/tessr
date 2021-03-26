import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import os
import numpy as np
import time
import math
from customDataset import CustomDataset
import runtimevariables as vars
from model import *

def set_device():
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device set to {device}')

def load_data(main_dir, transformations, train_split , batch_size = 32, is_test_set = False , num_workers = 0):
    
    ##calling the dataset class and making an instance of it
    dataset = CustomDataset(main_dir,transform = transformations , is_test_set=is_test_set)
    
    global num_classes, train_size , valid_size , dataset_size
    num_classes = dataset.get_num_classes()
    dataset_size = len(dataset)
    train_size = math.ceil(train_split * len(dataset))
    valid_size = len(dataset) - train_size
    train_set , valid_set = torch.utils.data.random_split(dataset,[train_size,valid_size])
    
    #creating train and validation loader
    train_loader = torch.utils.data.DataLoader(train_set , batch_size ,shuffle = True , num_workers = num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size ,shuffle = True, num_workers = num_workers )

    return train_loader , valid_loader

def adam(model , lr):
    return optim.Adam(model.parameters(), lr = lr)

def criterion():
    return nn.CrossEntropyLoss()

def lr_scheduler(optimizer, factor, patience, verbose):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=factor, patience=patience, verbose=verbose)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def train_model(model, train_loader, optimizer, loss_func, num_epochs, lr_scheduler, valid_loader=None):
    model
    #model.train()
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        loss_after_epoch, accuracy_after_epoch = 0, 0
        num_labels = 0
        for index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            preds = model(images)
            loss = loss_func(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_after_epoch += loss
            accuracy_after_epoch += get_num_correct(preds, labels)
            num_labels += torch.numel(labels)
        loss_after_epoch /= 100
        print(f'Epoch: {epoch}/{num_epochs}  Acc: {(accuracy_after_epoch/num_labels):.5f}  Loss: {loss_after_epoch:.5f} Duration: {(time.time()-start_time):.2f}s',
              end='  ' if valid_loader is not None else '\n')
        if valid_loader is not None:
            if epoch == num_epochs:
                validate_model(model, valid_loader, loss_func,
                               lr_scheduler, plot_cm=True)
            else:
                validate_model(model, valid_loader, loss_func,
                               lr_scheduler, plot_cm=False)
    return


def validate_model(model, valid_loader, loss_func, lr_scheduler, plot_cm=False):
    model.to
    model.eval()
    val_acc, val_loss = 0, 0
    true_labels, pred_labels = [], []
    num_labels = 0
    with torch.no_grad():
        for batch in valid_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            preds = model(images)
            val_loss += loss_func(preds, labels)
            val_acc += get_num_correct(preds, labels)
            num_labels += torch.numel(labels)
            if plot_cm:
                pred_labels += preds.argmax(dim=1)
                true_labels += labels

    val_acc /= num_labels
    print(f'Val_acc: {val_acc:.5f}  Val_loss: {(val_loss/100):.5f}')
    lr_scheduler.step(val_loss)
    return


def save_checkpoint(state, save_path, filename):
      print('Saving checkpoint...')
      save_path = os.path.join(save_path, filename)
      torch.save(state, save_path)

def get_checkpoint(checkpoint_path):
  print('Loading checkpoint...')
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  # epoch = checkpoint['epoch']
  # loss = checkpoint['loss']
  return model, optimizer#, epoch, loss

def early_stopping(measures, delta, patience):
   #pass
   count=0
   for measure in measures:
     measure_plus = measure + delta
     measure_minus = measure - delta
     for x in measures:
       if measure_minus<=x<=measure_plus:
         count +=1
def save_onnx(model):
    model.eval()
    #model.to('cpu')
    #model.features.set_swish(memory_efficient=False)
    input = torch.randn(2, 3, vars.image_size, vars.image_size)
    torch.onnx.export(model, input, vars.model_save_dir+'resnet.onnx',verbose=True)
    print('Model successfully saved in onnx format.')
    return

transform_list =transforms.Compose([transforms.Resize(vars.image_size), transforms.ToTensor(),
                           transforms.Normalize(vars.rgb_mean, vars.rgb_std)])
print('Setting device...')
set_device()
print('Creating training and validation dataloaders...')
train_loader, valid_loader = load_data(vars.root_dir, transformations=transform_list,batch_size=vars.batch_size, train_split=vars.train_split, num_workers=vars.num_workers)
print(f'Total of {num_classes} classes were found in the dataset!')
print('Defining model')
print('Initializing the optimizer...')
print('Training the model...')
if vars.load_checkpoint:
      model, optimizer = get_checkpoint(vars.checkpoint_path)
else:
  model = Resnet(input_channels=3, num_classes=num_classes,load_pretrained_weights=vars.load_pretrained_weights,train_only_last_layer=vars.train_only_last_layer)

optimizer = adam(model, vars.learning_rate)
loss_func = criterion()
scheduler = lr_scheduler(optimizer, factor=0.1, patience=3, verbose=2)

train_model(model ,train_loader, optimizer=optimizer, loss_func=loss_func,num_epochs=vars.num_epochs, lr_scheduler=scheduler, valid_loader=valid_loader)
#torch.save(model, '/content/drive/MyDrive/Inspiring/models/pollen-eff-b2.pt')
save_onnx(model)
