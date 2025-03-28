"""
Organize the TrashNet dataset into train and validation splits,
remapping categories as needed for our classifier.
"""

import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Define source and destination directories
source_dir = "data"
dest_dir = "data/trashnet-prepared"

# Define class mapping
class_mapping = {
    'plastic': 'plastic',
    'metal': 'metal',
    'paper': 'paper',
    'glass': 'glass',
    'cardboard': 'paper',  # Map cardboard to paper
    'trash': 'other'       # Map trash to other
}

# Create destination directories
os.makedirs(dest_dir, exist_ok=True)
os.makedirs(f"{dest_dir}/train", exist_ok=True)
os.makedirs(f"{dest_dir}/val", exist_ok=True)

# Create class directories in train and val
for target_class in set(class_mapping.values()):
    os.makedirs(f"{dest_dir}/train/{target_class}", exist_ok=True)
    os.makedirs(f"{dest_dir}/val/{target_class}", exist_ok=True)

# Process each class
for class_name, target_class in class_mapping.items():
    source_class_dir = f"{source_dir}/{class_name}"
    
    if not os.path.exists(source_class_dir):
        print(f"Warning: Class directory {source_class_dir} not found")
        continue
        
    # Get list of images
    images = [f for f in os.listdir(source_class_dir) if f.endswith('.jpg')]
    
    # Shuffle for random split
    random.shuffle(images)
    
    # Determine split point (80% training, 20% validation)
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Processing {class_name} -> {target_class}: {len(train_images)} train, {len(val_images)} val")
    
    # Copy training images
    for img in train_images:
        src = f"{source_class_dir}/{img}"
        dst = f"{dest_dir}/train/{target_class}/{img}"
        shutil.copy(src, dst)
    
    # Copy validation images
    for img in val_images:
        src = f"{source_class_dir}/{img}"
        dst = f"{dest_dir}/val/{target_class}/{img}"
        shutil.copy(src, dst)

print(f"Dataset organized at {dest_dir}")
print("Classes have been mapped as follows:")
for source, target in class_mapping.items():
    print(f"  {source} -> {target}")