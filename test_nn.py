import sys
import cv2  
import torch
import torch.nn as nn
from argparse import ArgumentParser
from downloaded_dataset.translate import translate
from resnet_model import ResNetModel


def get_args():
  parser = ArgumentParser(description="CNN training")
  parser.add_argument("--image_path", '-p', type=str)
  parser.add_argument("--image_size", '-i', type=int, default=224)
  parser.add_argument("--checkpoint", '-c', type=str, default="trained_models/best_resnet34.pt")
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = get_args()

  categories = ['pecora', 'elefante', 'scoiattolo', 'cane', 'farfalla', 
                'gatto', 'mucca', 'ragno', 'gallina', 'cavallo']

  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  model = ResNetModel(10, 3).to(device=device)
  if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["weights"])
  else:
    print("No check point!")
    exit(0)

  model.eval()
  ori_image = cv2.imread(args.image_path)
  image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (args.image_size, args.image_size))
  # image = cv2.transpose(image, (2, 0, 1)) / 255.0
  image = image.transpose((2, 0, 1)) / 255.0
  image = image[None, :, :, :] # 1x3x224x224
  image = torch.from_numpy(image).to(device).float()
  softmax = nn.Softmax(dim=1)

  with torch.no_grad():
    output = model(image)
    print(output)
    probs = softmax(output)
    print(probs)

  idx = torch.argmax(probs)
  predicted_class = translate[categories[idx]] 
  print("THe test image is about {} with confident score of {}"
        .format(predicted_class, probs[0, idx]))
  
  cv2.imshow("{}: {:.2f}%".format(predicted_class, probs[0, idx] * 100), ori_image)
  cv2.waitKey(0)