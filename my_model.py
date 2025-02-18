import torch
import torch.nn as nn
torch.cuda.empty_cache()

class SimpleNN(nn.Module):
  def __init__(self):
    super().__init__()
    num_classes = 10
    self.flatten = nn.Flatten()
    self.fc1 = nn.Sequential(
      nn.Linear(in_features=3*200*200, out_features=256),
      nn.ReLU()
    )
    self.fc2 = nn.Sequential(
      nn.Linear(in_features=256, out_features=512),
      nn.ReLU()
    )
    self.fc3 = nn.Sequential(
      nn.Linear(in_features=512, out_features=1024),
      nn.ReLU()
    )
    self.fc4 = nn.Sequential(
      nn.Linear(in_features=1024, out_features=512),
      nn.ReLU()
    )
    self.fc5 = nn.Sequential(
      nn.Linear(in_features=512, out_features=num_classes),
      nn.ReLU()
    )

  def forward(self, x):
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)
    x = self.fc5(x)
    return x
  
if __name__ == '__main__':
  model = SimpleNN()
  input_data = torch.rand(8, 3, 200, 200)
  if torch.cuda.is_available():
    model.cuda() # in_place function
    input_data = input_data.cuda()
  while True:
    result = model(input_data)
    print(result.shape)
