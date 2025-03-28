"""
Real-Time Recyclable Object Detection and Classification System

This project implements a computer vision system that:
1. Detects objects in real-time video streams
2. Classifies detected objects as recyclable or non-recyclable
3. Provides visual feedback through a user interface

The implementation uses YOLOv8 for object detection and a custom classification model
trained on recyclable materials. The system is optimized for real-time performance
on both standard computing systems and edge devices.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# For YOLOv8, we'll use the Ultralytics implementation
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics package not found. Installing...")
    os.system('pip install ultralytics')
    from ultralytics import YOLO

# Define the recyclable materials classifier
class RecyclableClassifier(nn.Module):
    """
    Custom classifier for recyclable materials using transfer learning
    with a pre-trained MobileNetV2 backbone.
    """
    def __init__(self, num_classes=6):  # Default to common waste categories: plastic, metal, paper, glass, organic, non-recyclable
        super(RecyclableClassifier, self).__init__()

        # Load pre-trained MobileNetV2 model with updated API
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Replace the last classifier layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Define the main RecyclableDetectionSystem class
class RecyclableDetectionSystem:
    """
    Main system class that integrates object detection and recyclable classification.
    """
    def __init__(self,
                 detector_model='yolov8n.pt',  # Use YOLOv8 nano by default (smallest)
                 classifier_weights='recyclable_classifier.pt',
                 device=None,
                 conf_threshold=0.25,
                 recyclable_categories=None):
        """
        Initialize the detection system.

        Args:
            detector_model: Path to YOLO model or model name
            classifier_weights: Path to trained classifier weights
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            conf_threshold: Confidence threshold for detections
            recyclable_categories: Dictionary mapping class indices to recyclability status
        """
        # Determine device (GPU if available, otherwise CPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load the object detector
        try:
            self.detector = YOLO(detector_model)
            print(f"Loaded detector: {detector_model}")
        except Exception as e:
            print(f"Error loading detector model: {e}")
            sys.exit(1)

        # Load the recyclable classifier
        self.classifier = RecyclableClassifier()

        if os.path.exists(classifier_weights):
            try:
                self.classifier.load_state_dict(torch.load(classifier_weights, map_location=self.device))
                print(f"Loaded classifier weights from: {classifier_weights}")
            except Exception as e:
                print(f"Error loading classifier weights: {e}")
                print("Using untrained classifier - please train before use.")
        else:
            print(f"Classifier weights not found at: {classifier_weights}")
            print("Using untrained classifier - please train before use.")

        self.classifier.to(self.device)
        self.classifier.eval()

        # Set confidence threshold
        self.conf_threshold = conf_threshold

        # Define recyclable categories
        # By default, we'll consider plastic, metal, paper, and glass as recyclable
        if recyclable_categories is None:
            self.recyclable_categories = {
                0: True,    # plastic - recyclable
                1: True,    # metal - recyclable
                2: True,    # paper - recyclable
                3: True,    # glass - recyclable
                4: False,   # organic - not recyclable (in standard recycling)
                5: False,   # non-recyclable
            }
        else:
            self.recyclable_categories = recyclable_categories

        # Define image transformation for classifier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Class names for classifier
        self.class_names = ['plastic', 'metal', 'paper', 'glass', 'organic', 'other']

    def classify_object(self, image_crop):
        """Classify a cropped object image as recyclable or not"""
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))

        # Apply transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        class_idx = predicted_idx.item()
        confidence = F.softmax(outputs, dim=1)[0][class_idx].item()

        # Determine if recyclable based on category
        is_recyclable = self.recyclable_categories.get(class_idx, False)
        material_type = self.class_names[class_idx]

        return {
            'is_recyclable': is_recyclable,
            'material_type': material_type,
            'class_idx': class_idx,
            'confidence': confidence
        }

    def process_frame(self, frame):
        """
        Process a single frame:
        1. Detect objects in the frame
        2. Classify each detected object
        3. Return annotated frame with detection and classification results
        """
        # Make a copy of the frame for annotations
        annotated_frame = frame.copy()

        # Run object detection on the frame
        detections = self.detector(frame, conf=self.conf_threshold)[0]

        # Process each detection
        for detection in detections.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = detection

            # Convert to integers for cropping
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Crop the detected object
            object_crop = frame[y1:y2, x1:x2]

            # Skip if crop is empty
            if object_crop.size == 0:
                continue

            # Classify the object as recyclable or not
            classification = self.classify_object(object_crop)

            # Draw bounding box (green for recyclable, red for non-recyclable)
            color = (0, 255, 0) if classification['is_recyclable'] else (0, 0, 255)

            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            class_name = detections.names[int(class_id)]
            material = classification['material_type']
            recyclable_text = "Recyclable" if classification['is_recyclable'] else "Non-recyclable"
            label = f"{class_name}: {material} ({recyclable_text}) {conf:.2f}"

            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame

    def run_on_video(self, source=0, output=None, show=True):
        """
        Run the detection and classification on a video source.

        Args:
            source: Video source (0 for webcam, or path to video file)
            output: Path to save output video (None for no output)
            show: Whether to display the output in a window
        """
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize video writer if output path is provided
        writer = None
        if output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            start_time = time.time()
            processed_frame = self.process_frame(frame)
            inference_time = time.time() - start_time

            # Add FPS information to the frame
            fps_text = f"FPS: {1.0 / max(inference_time, 0.001):.1f}"
            cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write to output if specified
            if writer:
                writer.write(processed_frame)

            # Display the processed frame
            if show:
                cv2.imshow("Recyclable Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

# Training module for the recyclable classifier
def train_classifier(data_dir, output_path, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train the recyclable classifier on a dataset.

    Args:
        data_dir: Directory containing training data organized by class
        output_path: Path to save trained model weights
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    # Define data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create data directory paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Check if data directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Data directories not found at {train_dir} and {val_dir}")
        print("Please organize your data as follows:")
        print(f"{data_dir}/train/")
        print(f"  ├── plastic/")
        print(f"  ├── metal/")
        print(f"  ├── paper/")
        print(f"  ├── glass/")
        print(f"  ├── organic/")
        print(f"  └── other/")
        print(f"{data_dir}/val/ (same structure)")
        return

    # Load datasets
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Print class mapping
    print("Class mapping:")
    for class_idx, class_name in enumerate(train_dataset.classes):
        print(f"  {class_idx}: {class_name}")

    # Initialize model
    num_classes = len(train_dataset.classes)
    model = RecyclableClassifier(num_classes=num_classes)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model with val_acc: {val_acc:.4f}")

    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")

# Download TrashNet dataset utility using Hugging Face
def download_trashnet():
    """
    Download and prepare the TrashNet dataset for training using Hugging Face datasets.
    """
    from datasets import load_dataset
    from PIL import Image
    import shutil
    import random
    
    print("Downloading TrashNet dataset from Hugging Face...")
    
    # Create data directories
    dataset_dir = "data/trashnet-prepared"
    os.makedirs("data", exist_ok=True)
    os.makedirs(f"{dataset_dir}/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/val", exist_ok=True)
    
    # Define class mapping
    class_mapping = {
        'plastic': 'plastic',
        'metal': 'metal',
        'paper': 'paper',
        'glass': 'glass',
        'cardboard': 'paper',  # Map cardboard to paper
        'trash': 'other'       # Map trash to other
    }
    
    # Get unique target classes
    target_classes = set(class_mapping.values())
    
    # Create class directories in train and val
    for class_name in target_classes:
        os.makedirs(f"{dataset_dir}/train/{class_name}", exist_ok=True)
        os.makedirs(f"{dataset_dir}/val/{class_name}", exist_ok=True)
    
    try:
        # Load the dataset from Hugging Face
        print("Loading the TrashNet dataset...")
        dataset = load_dataset("garythung/trashnet")
        
        print("Dataset loaded. Organizing for training...")
        
        # Since the dataset structure may not include labels,
        # we'll create a simple synthetic classifier for demonstration
        
        # Define some colors for each category to create synthetic data
        # when labels aren't available in the dataset
        colors = {
            'plastic': (255, 200, 200),  # light red
            'metal': (200, 200, 255),    # light blue
            'paper': (255, 255, 200),    # light yellow
            'glass': (200, 255, 255),    # light cyan
            'other': (220, 220, 220)     # light gray
        }
        
        # Extract the train split
        train_data = dataset["train"]
        
        # Split the data 80% for training, 20% for validation
        total_samples = len(train_data)
        train_size = int(0.8 * total_samples)
        
        indices = list(range(total_samples))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        print(f"Total samples: {total_samples}")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        
        # Function to classify an image based on average color
        # This is a simple heuristic when real labels aren't available
        def classify_by_color(image):
            img_array = np.array(image)
            avg_color = np.mean(img_array, axis=(0, 1))
            
            # Find closest color in our predefined colors
            min_dist = float('inf')
            best_class = 'other'
            
            for class_name, color in colors.items():
                dist = sum((a - b) ** 2 for a, b in zip(avg_color, color))
                if dist < min_dist:
                    min_dist = dist
                    best_class = class_name
            
            return best_class
        
        # Process and save training images
        print("Processing training images...")
        for idx in train_indices:
            # Get image
            image = train_data[idx]['image']
            
            # If the dataset has labels, use them
            # Otherwise use our color classifier
            if 'label' in train_data[idx]:
                labels = ['plastic', 'metal', 'paper', 'glass', 'cardboard', 'trash']
                label_idx = train_data[idx]['label']
                class_name = labels[label_idx]
            else:
                # Use our simple classifier based on color
                class_name = classify_by_color(image)
            
            # Map to target class
            target_class = class_mapping.get(class_name, 'other')
            
            # Save the image
            image_path = f"{dataset_dir}/train/{target_class}/{class_name}_{idx}.jpg"
            image.save(image_path)
        
        # Process and save validation images
        print("Processing validation images...")
        for idx in val_indices:
            # Get image
            image = train_data[idx]['image']
            
            # If the dataset has labels, use them
            # Otherwise use our color classifier
            if 'label' in train_data[idx]:
                labels = ['plastic', 'metal', 'paper', 'glass', 'cardboard', 'trash']
                label_idx = train_data[idx]['label']
                class_name = labels[label_idx]
            else:
                # Use our simple classifier based on color
                class_name = classify_by_color(image)
            
            # Map to target class
            target_class = class_mapping.get(class_name, 'other')
            
            # Save the image
            image_path = f"{dataset_dir}/val/{target_class}/{class_name}_{idx}.jpg"
            image.save(image_path)
            
        print(f"Dataset prepared at {dataset_dir}")
        print("Classes have been mapped as follows:")
        print("  plastic -> plastic")
        print("  metal -> metal")
        print("  paper -> paper")
        print("  glass -> glass")
        print("  cardboard -> paper (merged with paper)")
        print("  trash -> other")
        
        return dataset_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main function to run the application
def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Real-time Recyclable Object Detection and Classification System")
    parser.add_argument('--mode', choices=['train', 'detect', 'download_data'], default='detect',
                        help='Mode to run: train classifier, run detection, or download data')
    parser.add_argument('--source', default=0,
                        help='Source for detection (0 for webcam, or path to video file)')
    parser.add_argument('--output', default=None,
                        help='Path to save output video (None for no output)')
    parser.add_argument('--detector', default='yolov8n.pt',
                        help='Path to YOLOv8 detector model')
    parser.add_argument('--classifier', default='recyclable_classifier.pt',
                        help='Path to trained classifier weights')
    parser.add_argument('--data_dir', default='data/trashnet-prepared',
                        help='Directory containing training data (for training mode)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (for training mode)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (for training mode)')
    parser.add_argument('--device', default=None,
                        help='Device to run on (None for auto, cpu, or cuda)')

    args = parser.parse_args()

    if args.mode == 'download_data':
        # Download and prepare the TrashNet dataset
        download_trashnet()

    elif args.mode == 'train':
        # Train the recyclable classifier
        train_classifier(
            data_dir=args.data_dir,
            output_path=args.classifier,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    elif args.mode == 'detect':
        # Initialize the detection system
        system = RecyclableDetectionSystem(
            detector_model=args.detector,
            classifier_weights=args.classifier,
            device=args.device
        )

        # Run detection on video source
        system.run_on_video(
            source=args.source,
            output=args.output,
            show=True
        )

if __name__ == "__main__":
    main()
