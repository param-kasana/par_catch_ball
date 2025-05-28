import os
import shutil
import random

source_images = 'dataset_frames'
output_images = 'your_dataset/images'
output_labels = 'your_dataset/labels'

os.makedirs(f'{output_images}/train', exist_ok=True)
os.makedirs(f'{output_images}/val', exist_ok=True)
os.makedirs(f'{output_labels}/train', exist_ok=True)
os.makedirs(f'{output_labels}/val', exist_ok=True)

# Get all image files
images = [f for f in os.listdir(source_images) if f.endswith('.jpg')]
random.shuffle(images)

split_ratio = 0.8
split_index = int(len(images) * split_ratio)
train_images = images[:split_index]
val_images = images[split_index:]

def copy_files(image_list, split):
    for img in image_list:
        label = img.replace('.jpg', '.txt')
        shutil.copy(os.path.join(source_images, img), os.path.join(output_images, split, img))
        shutil.copy(os.path.join(source_images, label), os.path.join(output_labels, split, label))

copy_files(train_images, 'train')
copy_files(val_images, 'val')

print("✅ Dataset split into train and val folders.")
