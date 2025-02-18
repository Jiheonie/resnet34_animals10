import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose 
from torch.utils.data import DataLoader

class AnimalDataset(Dataset):
  def __init__(self, root, train=True, transform=None):
    self.root = root
    self.transform = transform
    if train:
      mode = 'train'
    else:
      mode = 'test'
    self.root = os.path.join(root, mode)
    # print(os.listdir(self.root))
    self.categories = ['pecora', 'elefante', 'scoiattolo', 'cane', 'farfalla', 
                        'gatto', 'mucca', 'ragno', 'gallina', 'cavallo']
    self.image_paths = []
    self.labels = []
    for i, category in enumerate(self.categories):
      category_path = os.path.join(self.root, category)
      for file_path in os.listdir(category_path):
        file_path = os.path.join(category_path, file_path)
        self.image_paths.append(file_path)
        self.labels.append(i)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path).convert("RGB")
    # khi nào chỉ show 1 image thì dùng, load dataloader mà dùng, lag  
    # image.show() 
    if self.transform:
      image = self.transform(image)
    label = self.labels[idx]
    return image, label


if __name__ == '__main__':
  root = 'mine/animals/dataset'

  transform = Compose([
    Resize((200, 200)),
    # other steps
    ToTensor()
  ])

  dataset = AnimalDataset(root, train=True, transform=transform)
  image, label = dataset.__getitem__(1913)
  print(np.asarray(image).shape)
  print(image)
  print(label)

  training_dataloader = DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=True
  )

  i = 0
  for images, labels in training_dataloader:
    print(images.shape)
    print(labels)
    i += 1
    if i == 5:
      break