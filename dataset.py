import os
import random
import shutil

dataset_dir = 'mine/animals/2/raw-img/'
output_dir = 'mine/animals/dataset/'

train_ratio = 0.8

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(dataset_dir):
  print(".")
  class_path = os.path.join(dataset_dir, class_name)
  
  if not os.path.isdir(class_path):
    continue  # Skip if not a directory
  
  images = os.listdir(class_path)
  random.shuffle(images)
  
  split_idx = int(len(images) * train_ratio)
  train_images = images[:split_idx]
  test_images = images[split_idx:]

  # Create class subdirectories in train and test folders
  os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
  os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

  # Move images
  for img in train_images:
    shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
  
  for img in test_images:
    shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

print("Dataset split complete!")