# Download TrashNet dataset utility - Creating synthetic data
def download_trashnet():
    """
    Create synthetic dataset for training the recyclable classifier.
    """
    import os
    import random
    from PIL import Image
    
    print("Creating synthetic dataset for training...")
    
    # Create data directories
    dataset_dir = "data/trashnet-prepared"
    os.makedirs("data", exist_ok=True)
    os.makedirs(f"{dataset_dir}/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/val", exist_ok=True)
    
    # Define target classes
    target_classes = ['plastic', 'metal', 'paper', 'glass', 'other']
    
    # Create class directories in train and val
    for class_name in target_classes:
        os.makedirs(f"{dataset_dir}/train/{class_name}", exist_ok=True)
        os.makedirs(f"{dataset_dir}/val/{class_name}", exist_ok=True)
    
    try:
        print("Generating synthetic training data...")
        
        # Define some simple colors for each class
        class_colors = {
            'plastic': (255, 200, 200),  # light red
            'metal': (200, 200, 255),    # light blue
            'paper': (255, 255, 200),    # light yellow
            'glass': (200, 255, 255),    # light cyan
            'other': (220, 220, 220)     # light gray
        }
        
        # Generate the images
        for target_class, color in class_colors.items():
            print(f"Generating images for class: {target_class}")
            
            # Create training images
            for i in range(50):
                # Create a 224x224 image with the class color
                img = Image.new('RGB', (224, 224), color)
                
                # Add some random noise to make each image unique
                for x in range(224):
                    for y in range(224):
                        if random.random() < 0.1:  # 10% chance of noise
                            r, g, b = color
                            r = max(0, min(255, r + random.randint(-20, 20)))
                            g = max(0, min(255, g + random.randint(-20, 20)))
                            b = max(0, min(255, b + random.randint(-20, 20)))
                            img.putpixel((x, y), (r, g, b))
                
                # Save the image
                img.save(f"{dataset_dir}/train/{target_class}/{target_class}_{i}.jpg")
            
            # Create validation images
            for i in range(20):
                # Create a 224x224 image with the class color
                img = Image.new('RGB', (224, 224), color)
                
                # Add some random noise to make each image unique
                for x in range(224):
                    for y in range(224):
                        if random.random() < 0.1:  # 10% chance of noise
                            r, g, b = color
                            r = max(0, min(255, r + random.randint(-20, 20)))
                            g = max(0, min(255, g + random.randint(-20, 20)))
                            b = max(0, min(255, b + random.randint(-20, 20)))
                            img.putpixel((x, y), (r, g, b))
                
                # Save the image
                img.save(f"{dataset_dir}/val/{target_class}/{target_class}_{i}.jpg")
        
        print(f"Dataset prepared at {dataset_dir}")
        print("Classes generated:")
        print("  plastic")
        print("  metal")
        print("  paper")
        print("  glass")
        print("  other")
        
        return dataset_dir
        
    except Exception as e:
        print(f"Error generating dataset: {e}")
        return None

# Main function to run the application