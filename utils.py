import torch

def get_best_epoch():
  checkpoint = torch.load('trained_models/best_resnet34.pt')
  return checkpoint['epoch'], checkpoint['best_acc']

if __name__ == '__main__':
  epoch, best_acc = get_best_epoch()
  print(epoch)
  print(best_acc)