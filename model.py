import torch.nn as nn
from torchvision import datasets, models, transforms


class Resnet(nn.Module):
  def __init__(self,input_channels, num_classes,load_pretrained_weights=True, train_only_last_layer=False):
    super(Resnet, self).__init__()
    self.input_channels = input_channels
    self.num_classes = num_classes
    self.load_pretrained_weights = load_pretrained_weights
    self.train_only_last_layer = train_only_last_layer
    
    if self.load_pretrained_weights:
      self.features = models.resnet50(pretrained=True)
    else:
      self.features = models.resnet50(pretrained = False)
    
    if self.train_only_last_layer:
      for param in self.features.parameters():
        param.requires_grad = False

    in_ftrs = self.features.fc.in_features
    
    #print(in_ftrs)

    self.features.fc = nn.Sequential(
            nn.Linear(in_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512 , self.num_classes))


  def forward(self, inputs):
    x = self.features(inputs)
    
    return x
