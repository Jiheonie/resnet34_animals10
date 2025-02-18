import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from my_dataset import AnimalDataset
from my_model import SimpleNN
from resnet_model import ResNetModel
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD
from torchvision.transforms import ToTensor, Resize, Compose 
import torch
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_args():
  parser = ArgumentParser(description="CNN training")
  parser.add_argument("--root", '-r', type=str, default='dataset')
  parser.add_argument("--epochs", '-e', type=int, default=500, help="Num of epoch")
  parser.add_argument("--batch_size", '-b', type=int, default=64, help="Batch size")
  parser.add_argument("--num_classes", '-n', type=int, default=10)
  parser.add_argument("--image_size", '-i', type=int, default=224)
  parser.add_argument("--logging", '-l', type=str, default="tensorboard", help="Logging")
  parser.add_argument("--trained_models", '-m', type=str, default="trained_models")
  parser.add_argument("--checkpoint", '-c', type=str, default=None)
  args = parser.parse_args()
  return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
  figure = plt.figure(figsize=(20, 20))
  plt.imshow(cm, interpolation="nearest", cmap="OrRd")
  plt.title("Confusion Matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
  threshold = cm.max() / 2

  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      color = "white" if cm[i, j] > threshold else "black"
      plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  writer.add_figure('confusion_matrix', figure, epoch)


if __name__ == '__main__':
  # -----------------------check if device has gpu-------------------
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  # -----------------------get arguments----------------------------
  args = get_args()
  print("Number of Epochs: {}".format(args.epochs))
  print("Batch size: {}".format(args.batch_size))

  # -----------------------load dataset------------------------------
  transform = Compose([
    Resize((args.image_size, args.image_size)),
    # other steps
    ToTensor()
  ])
  train_dataset = AnimalDataset(root=args.root, train=True, transform=transform)
  train_dataloader = DataLoader(
    dataset=train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=4,
    drop_last=True
  )
  test_dataset = AnimalDataset(root=args.root, train=False, transform=transform)
  test_dataloader = DataLoader(
    dataset=test_dataset, 
    batch_size=args.batch_size, 
    shuffle=False,
    num_workers=4,
    drop_last=False
  )

  # -------------------------------------------------------
  # if os.path.isdir(args.logging):
    # shutil.rmtree(args.logging)
  if not os.path.isdir(args.trained_models):
    os.mkdir(args.trained_models)

  # -----------------------create tensorboard-----------------------------
  writer = SummaryWriter(args.logging)

  # ------------------------define models and load weights---------------------
  model = ResNetModel(args.num_classes, 3).to(device=device)
  criterion = nn.CrossEntropyLoss()
  optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9) # momentum min = 0.9
  if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_acc"]
    model.load_state_dict(checkpoint["weights"])
    optimizer.load_state_dict(checkpoint["optimizer"])
  else:
    start_epoch = 0
    best_acc = 0

  # --------------------------
  num_iters = len(train_dataloader)
  for epoch in range(start_epoch, args.epochs):
    # ----------------------------train----------------------------------
    model.train()
    progress_bar = tqdm(train_dataloader, colour='cyan')
    for iter, (images, labels) in enumerate(progress_bar):
      images = images.to(device)
      labels = labels.to(device)

      # forward
      outputs = model(images) 
      loss_value = criterion(outputs, labels)
      progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}"
        .format(epoch + 1, args.epochs, iter + 1, num_iters, loss_value))
      writer.add_scalar("Train/Loss", loss_value, epoch * num_iters + iter)

      # backward and optimize
      optimizer.zero_grad() # khong can tich luy gradient => lam sach buffer
      loss_value.backward()
      optimizer.step()

    # -----------------------------validate-------------------------------
    model.eval()
    all_predictions = []
    all_labels = []
    for iter, (images, labels) in enumerate(test_dataloader):
      all_labels.extend(labels)
      if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
      
      with torch.no_grad():
        predictions = model(images)  # prediction shape 64x10
        indices = torch.argmax(predictions.cpu(), dim=1) # give predictions from gpu to cpu
        all_predictions.extend(indices)
        loss_value = criterion(predictions, labels)

    all_labels = [label.item() for label in all_labels]
    all_predictions = [prediction.item() for prediction in all_predictions]
    plot_confusion_matrix(writer, 
                          confusion_matrix(all_labels, all_predictions), 
                          test_dataset.categories, epoch)
    accuracy = accuracy_score(all_labels, all_predictions)

    print("Epoch {}/{}: Accuracy: {}".format(epoch + 1, args.epochs, accuracy))
    writer.add_scalar("Val/Accuracy", accuracy, epoch)
    # print(classification_report(all_labels, all_predictions))


    # ------------------- save model -------------------------------
    checkpoint = {
      "epoch": epoch + 1,
      "best_acc": best_acc,
      "weights": model.state_dict(),
      "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, "{}/last_resnet34.pt".format(args.trained_models))
    if accuracy > best_acc:
      torch.save(checkpoint, "{}/best_resnet34.pt".format(args.trained_models))
      best_acc = accuracy