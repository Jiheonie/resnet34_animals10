import torch 
import torch.nn as nn

class Residual(torch.nn.Module):
  def __init__(self, module: torch.nn.Module, projection: torch.nn.Module = None):
    super().__init__()
    self.module = module
    self.projection = projection
  
  def forward(self, x):
    output = self.module(x)
    if self.projection is not None:
      x = self.projection(x)
    return output + x


class Block(nn.Module):
  def __init__(self, in_channels, out_channels, k=3, s=1, padding=1):
    super().__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, k, s, padding=padding),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU()
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(out_channels, out_channels, k, 1, padding=padding),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU()
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x
    

class ResNetModel(nn.Module):
  def __init__(self, num_classes, in_channels):
    super().__init__()
    self.stage1 = nn.Sequential(
      nn.Conv2d(in_channels, 64, 7, 2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
    )
    self.stage2 = nn.Sequential(
      nn.MaxPool2d(3, 2, padding=1),
      Block(64, 64, 3, 1),
      Block(64, 64, 3, 1),
      Block(64, 64, 3, 1),
    )
    self.stage3 = nn.Sequential(
      Block(64, 128, 3, 2),
      Residual(Block(128, 128, 3, 1)),
      Residual(Block(128, 128, 3, 1)),
      Residual(Block(128, 128, 3, 1)),
    )
    self.stage4 = nn.Sequential(
      Block(128, 256, 3, 2),
      Residual(Block(256, 256, 3, 1)),
      Residual(Block(256, 256, 3, 1)),
      Residual(Block(256, 256, 3, 1)),
      Residual(Block(256, 256, 3, 1)),
      Residual(Block(256, 256, 3, 1)),
    )
    self.stage5 = nn.Sequential(
      Block(256, 512, 3, 2),
      Residual(Block(512, 512, 3, 1)),
      Residual(Block(512, 512, 3, 1)),
    )
    self.avg_pool = nn.AvgPool2d(7)
    self.fc = nn.Sequential(
      nn.Linear(in_features=512, out_features=num_classes),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.stage1(x)
    # print(x.shape)
    x = self.stage2(x)
    # print(x.shape)
    x = self.stage3(x)
    # print(x.shape)
    x = self.stage4(x)
    # print(x.shape)
    x = self.stage5(x)
    # print(x.shape)
    x = self.avg_pool(x).view(x.shape[0], -1)
    # print(x.shape)
    x = self.fc(x)
    # print(x.shape)
    return x


if __name__ == '__main__':
  model = ResNetModel(10, 3)
  input_data = torch.rand(200, 3, 224, 224)
  # if torch.cuda.is_available():
  #   model.cuda()
  #   input_data = input_data.cuda()
  result = model(input_data)
  print(result)