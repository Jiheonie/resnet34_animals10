import os
import shutil
import kagglehub

# Download latest version
source = kagglehub.dataset_download("alessiocorrado99/animals10")
print("Path to dataset files:", source)
destination = shutil.move(source, './')
os.rename(destination, './downloaded_dataset1')