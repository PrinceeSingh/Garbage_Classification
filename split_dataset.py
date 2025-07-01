import os
import shutil
import random

source_dir = 'garbage_dataset'  # Update this if your folder has a different name
target_dir = 'data'
split_ratio = 0.8  # 80% train, 20% val

# Make train/ and val/ folders
for split in ['train', 'val']:
    for class_name in os.listdir(source_dir):
        os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)

# Split images
for class_name in os.listdir(source_dir):
    files = os.listdir(os.path.join(source_dir, class_name))
    random.shuffle(files)
    split_point = int(split_ratio * len(files))
    train_files = files[:split_point]
    val_files = files[split_point:]

    for f in train_files:
        shutil.copy(os.path.join(source_dir, class_name, f),
                    os.path.join(target_dir, 'train', class_name, f))

    for f in val_files:
        shutil.copy(os.path.join(source_dir, class_name, f),
                    os.path.join(target_dir, 'val', class_name, f))

print("Dataset successfully split into train/ and val/ folders.")