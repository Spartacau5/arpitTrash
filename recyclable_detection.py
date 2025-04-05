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
import csv
from datetime import datetime

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
    def __init__(self, num_classes=5):  # Default to our categories: plastic, metal, paper, glass, other
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
                 debug_mode=False,
                 recyclable_categories=None,
                 csv_output=None):
        """
        Initialize the detection system.

        Args:
            detector_model: Path to YOLO model or model name
            classifier_weights: Path to trained classifier weights
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            conf_threshold: Confidence threshold for detections
            debug_mode: Show detailed material classification scores
            recyclable_categories: Dictionary mapping class indices to recyclability status
            csv_output: Path to save detection results in CSV format
        """
        # Determine device (GPU if available, otherwise CPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Debug mode for material classification debugging
        self.debug_mode = debug_mode

        # Load the object detector
        try:
            self.detector = YOLO(detector_model)
            print(f"Loaded detector: {detector_model}")
        except Exception as e:
            print(f"Error loading detector model: {e}")
            sys.exit(1)

        # Define potential waste objects from YOLO's classes
        # These are objects that could be discarded and classified as recyclable or trash
        self.potential_waste_objects = [
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
            'hot dog', 'pizza', 'donut', 'cake', 'book', 'vase', 'scissors',
            'teddy bear', 'toothbrush', 'backpack', 'umbrella', 'handbag', 'box',
            'tie', 'suitcase', 'paper', 'cardboard', 'can', 'bag', 'package',
            'plastic', 'cardboard', 'newspaper', 'magazine', 'glasses', 'eyeglasses',
            'spectacles'
        ]
        
        # Define common YOLO classes that could be recyclable materials
        # Maps YOLO class names to our material categories
        self.recyclable_item_map = {
            'bottle': 'plastic',     # Most bottles are plastic
            'wine glass': 'glass',   # Wine glasses are glass
            'cup': 'paper',          # Many cups are paper
            'bowl': 'paper',         # Many bowls can be paper
            'book': 'paper',         # Books are paper
            'vase': 'glass',         # Vases are often glass
            'scissors': 'metal',     # Scissors are metal
            'fork': 'metal',         # Forks are metal 
            'knife': 'metal',        # Knives are metal
            'spoon': 'metal',        # Spoons are metal
            'paper': 'paper',        # Paper is paper
            'cardboard': 'paper',    # Cardboard is classified as paper
            'box': 'paper',          # Most boxes are cardboard/paper
            'can': 'metal',          # Cans are metal
            'newspaper': 'paper',    # Newspapers are paper
            'magazine': 'paper'      # Magazines are paper
        }
        
        # Define specifically non-recyclable waste items (trash)
        self.trash_items = {
            'cell phone': 'e-waste',     # Electronic waste
            'mouse': 'e-waste',          # Electronic waste
            'keyboard': 'e-waste',       # Electronic waste
            'remote': 'e-waste',         # Electronic waste
            'vape': 'e-waste',           # Electronic waste
            'electronic cigarette': 'e-waste', # Electronic waste
            'battery': 'e-waste',        # Electronic waste
            'computer': 'e-waste',       # Electronic waste
            'laptop': 'e-waste',         # Electronic waste
            'monitor': 'e-waste',        # Electronic waste
            'backpack': 'mixed',         # Backpacks are often mixed materials
            'handbag': 'mixed',          # Handbags are often mixed materials
            'suitcase': 'mixed',         # Suitcases are often mixed materials
            'umbrella': 'mixed',         # Umbrellas are complex items
            'tie': 'fabric',             # Ties are fabric
            'teddy bear': 'plush',       # Teddy bears are not easily recyclable
            'styrofoam': 'non-recyclable', # Styrofoam is typically not recyclable
            'diaper': 'non-recyclable',  # Diapers are not recyclable
            'wrapper': 'non-recyclable', # Food wrappers are usually not recyclable
            'cigarette': 'non-recyclable', # Cigarette butts are not recyclable
            'lightbulb': 'specialized'   # Requires specialized recycling
        }
        
        # Define product categories that override material-based recyclability
        self.non_recyclable_categories = {
            'e-waste': "Electronic waste requires special disposal",
            'mixed': "Mixed materials are typically not recyclable",
            'fabric': "Fabrics require textile recycling",
            'plush': "Plush items are not recyclable",
            'non-recyclable': "This item is not recyclable",
            'specialized': "Requires specialized recycling"
        }
        
        # Food waste items (not recyclable, but compostable)
        self.food_waste = [
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'fruit', 'vegetable'
        ]

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
                0: True,    # glass - recyclable
                1: True,    # metal - recyclable
                2: False,   # other - not recyclable
                3: True,    # paper - recyclable
                4: True,    # plastic - recyclable
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

        # Class names for classifier - must match order from training dataset
        self.class_names = ['glass', 'metal', 'other', 'paper', 'plastic']

        # CSV output
        self.csv_output = csv_output
        self.csv_file = None
        self.csv_writer = None
        
        if self.csv_output:
            csv_exists = os.path.exists(self.csv_output)
            self.csv_file = open(self.csv_output, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            if not csv_exists:
                # Write header if the file is new
                self.csv_writer.writerow(['Timestamp', 'Object', 'Material', 'Confidence', 'Category', 'Is Recyclable'])

        # Add keywords that help identify e-waste and special handling products
        self.product_keywords = {
            'electronic': 'e-waste',
            'battery': 'e-waste',
            'charger': 'e-waste',
            'adapter': 'e-waste',
            'cable': 'e-waste',
            'wire': 'e-waste',
            'phone': 'e-waste',
            'computer': 'e-waste',
            'tablet': 'e-waste',
            'tv': 'e-waste',
            'vape': 'e-waste',
            'cigarette': 'non-recyclable',
            'styrofoam': 'non-recyclable',
            'foam': 'non-recyclable',
            'hazardous': 'specialized',
            'toxic': 'specialized',
            'chemical': 'specialized',
            'medicine': 'specialized',
            'pill': 'specialized',
            'drug': 'specialized',
            'makeup': 'specialized',
            'cosmetic': 'specialized',
            'bulb': 'specialized',
            'lamp': 'specialized',
            'toy': 'mixed',
            'mixed': 'mixed',
            'composite': 'mixed',
        }

        # Update trash items with more specific designations
        self.trash_items.update({
            # Electronics and batteries
            'cell phone': 'e-waste',
            'smartphone': 'e-waste',
            'laptop': 'e-waste',
            'tablet': 'e-waste',
            'desktop': 'e-waste',
            'monitor': 'e-waste',
            'tv': 'e-waste',
            'television': 'e-waste',
            'camera': 'e-waste',
            'printer': 'e-waste',
            'battery': 'e-waste',
            'charger': 'e-waste',
            'adapter': 'e-waste',
            'cable': 'e-waste',
            'cord': 'e-waste',
            'headphones': 'e-waste',
            'earbuds': 'e-waste',
            'speaker': 'e-waste',
            'microphone': 'e-waste',
            'clock': 'e-waste',
            'watch': 'e-waste',
            'calculator': 'e-waste',
            
            # Hazardous items
            'paint': 'hazardous',
            'pesticide': 'hazardous',
            'cleaner': 'hazardous',
            'chemical': 'hazardous',
            'medicine': 'hazardous',
            'pills': 'hazardous',
            'medication': 'hazardous',
            'syringe': 'hazardous',
            'needle': 'hazardous',
            'toner': 'hazardous',
            'ink': 'hazardous',
            'bleach': 'hazardous',
            'spray can': 'hazardous',
            'aerosol': 'hazardous',
            'propane': 'hazardous',
            'oil': 'hazardous',
            
            # Mixed/composite materials
            'blister pack': 'mixed',
            'toy': 'mixed',
            'shoes': 'mixed',
            'clothing': 'fabric',
            'textile': 'fabric',
            'furniture': 'mixed',
            'carpet': 'mixed',
            'rug': 'mixed',
            'mattress': 'mixed',
            'pillow': 'mixed',
            'hose': 'mixed',
            'pipe': 'mixed',
            'wire': 'mixed',
            'insulation': 'specialized',
            
            # Non-recyclable plastics
            'cling wrap': 'non-recyclable',
            'plastic wrap': 'non-recyclable',
            'plastic bag': 'plastic-film',
            'chip bag': 'non-recyclable',
            'candy wrapper': 'non-recyclable',
            'wrapper': 'non-recyclable',
            'styrofoam': 'non-recyclable',
            'foam': 'non-recyclable',
            
            # Special recycling programs
            'light bulb': 'specialized',
            'lightbulb': 'specialized',
            'fluorescent': 'specialized',
            'led bulb': 'specialized',
            'glasses': 'specialized',
            'eyeglasses': 'specialized',
        })
        
        # Expand non-recyclable categories with more information
        self.non_recyclable_categories.update({
            'e-waste': "Electronic waste requires special disposal at e-waste collection centers",
            'hazardous': "Hazardous materials require special disposal at hazardous waste facilities",
            'plastic-film': "Plastic film may be recyclable at special drop-off locations, not in regular recycling",
            'fabric': "Textiles should be donated or taken to textile recycling centers",
            'mixed': "Mixed material products are difficult to recycle in standard programs",
        })

        # Disposal instructions for different categories
        self.disposal_instructions = {
            'recyclable': "✅ Place in regular recycling bin",
            'trash': "🗑️ Place in regular trash bin",
            'compostable': "🍂 Place in compost bin",
            'e-waste': "🔌 Take to e-waste collection center",
            'hazardous': "⚠️ Take to hazardous waste facility",
            'specialized': "♻️ Requires special recycling program",
            'fabric': "👕 Donate or take to textile recycling",
            'plastic-film': "💼 Take to plastic film drop-off (some grocery stores)",
            'mixed': "⚠️ Check local recycling guidelines",
            'non-recyclable': "🗑️ Place in regular trash bin"
        }

    def classify_object(self, image_crop):
        """Classify a cropped object image as recyclable or not"""
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))

        # Apply transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            softmax_outputs = F.softmax(outputs, dim=1)[0]
            
            # Store all material confidences for debug mode
            material_confidences = {}
            for i, material in enumerate(self.class_names):
                material_confidences[material] = softmax_outputs[i].item()
            
            # Get top 2 predictions to see if there's ambiguity
            top_values, top_indices = torch.topk(softmax_outputs, 2)
            top1_idx, top2_idx = top_indices.tolist()
            top1_val, top2_val = top_values.tolist()
            
            # Compute advanced visual features to enhance material detection
            try:
                # Convert to different color spaces for analysis
                gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
                hsv_img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
                lab_img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2LAB)
                
                # Extract basic texture features
                gray_std = np.std(gray)                         # Reflectivity/texture roughness
                gray_mean = np.mean(gray)                       # Overall brightness
                
                # Extract color features
                saturation = np.mean(hsv_img[:,:,1]) / 255      # Color intensity
                brightness = np.mean(hsv_img[:,:,2]) / 255      # Brightness
                color_std = np.std(hsv_img[:,:,0]) / 255        # Color uniformity
                
                # Extract edge features for texture analysis
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
                
                # Use gradient for reflective properties (for glass, metal)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(sobelx**2 + sobely**2)
                gradient_mean = np.mean(gradient_mag)
                gradient_std = np.std(gradient_mag)
                
                # Extract features from LAB color space (good for material differentiation)
                l_channel = lab_img[:,:,0]
                a_channel = lab_img[:,:,1]
                b_channel = lab_img[:,:,2]
                l_std = np.std(l_channel)              # Lightness variation
                a_std = np.std(a_channel)              # Red-green variation
                b_std = np.std(b_channel)              # Blue-yellow variation
                
                # Calculate GLCM (Gray Level Co-occurrence Matrix) texture features
                # This helps distinguish between different material textures
                if gray.shape[0] > 10 and gray.shape[1] > 10:
                    # Resize for faster processing if needed
                    if gray.shape[0] > 100 or gray.shape[1] > 100:
                        small_gray = cv2.resize(gray, (100, 100))
                    else:
                        small_gray = gray
                        
                    # Reduce gray levels for GLCM computation
                    small_gray = (small_gray / 32).astype(np.uint8)
                    
                    # Calculate co-occurrence matrix
                    h, w = small_gray.shape
                    glcm_h = np.zeros((8, 8))
                    glcm_v = np.zeros((8, 8))
                    
                    # Horizontal co-occurrence
                    for i in range(h):
                        for j in range(w-1):
                            glcm_h[small_gray[i,j], small_gray[i,j+1]] += 1
                            
                    # Vertical co-occurrence  
                    for i in range(h-1):
                        for j in range(w):
                            glcm_v[small_gray[i,j], small_gray[i+1,j]] += 1
                            
                    # Normalize GLCMs
                    glcm_h = glcm_h / np.sum(glcm_h) if np.sum(glcm_h) > 0 else glcm_h
                    glcm_v = glcm_v / np.sum(glcm_v) if np.sum(glcm_v) > 0 else glcm_v
                    
                    # Calculate GLCM features
                    # Contrast - measures local variations
                    contrast_h = np.sum(np.square(np.arange(8) - np.arange(8).reshape(-1, 1)) * glcm_h)
                    contrast_v = np.sum(np.square(np.arange(8) - np.arange(8).reshape(-1, 1)) * glcm_v)
                    contrast = (contrast_h + contrast_v) / 2
                    
                    # Energy - measures textural uniformity
                    energy_h = np.sum(np.square(glcm_h))
                    energy_v = np.sum(np.square(glcm_v))
                    energy = (energy_h + energy_v) / 2
                    
                    # Homogeneity - measures closeness of element distribution
                    denom_h = 1 + np.square(np.arange(8) - np.arange(8).reshape(-1, 1))
                    denom_v = 1 + np.square(np.arange(8) - np.arange(8).reshape(-1, 1))
                    homogeneity_h = np.sum(glcm_h / denom_h)
                    homogeneity_v = np.sum(glcm_v / denom_v)
                    homogeneity = (homogeneity_h + homogeneity_v) / 2
                else:
                    contrast = 0
                    energy = 0
                    homogeneity = 0
                
                # Apply enhanced heuristics for material classification using all features
                
                # GLASS detection improvements
                is_glass = False
                if (brightness > 0.65 and saturation < 0.15 and 
                    gradient_std > 25 and edge_density < 0.1 and
                    energy > 0.25):
                    # High brightness, low saturation, high gradient variation
                    # These are typical for glass
                    is_glass = True
                    if top1_idx != 0 or top1_val < 0.7:  # 0 is glass class
                        top1_idx = 0
                        top1_val = max(top1_val, 0.75)
                        
                # METAL detection improvements
                is_metal = False
                if (gray_std > 40 and saturation < 0.25 and 
                    brightness > 0.4 and gradient_mean > 15 and
                    contrast > 0.4 and energy < 0.15):
                    # High texture variation, low saturation, high gradient
                    # These are typical for metal surfaces
                    is_metal = True
                    if top1_idx != 1 or top1_val < 0.7:  # 1 is metal class
                        top1_idx = 1
                        top1_val = max(top1_val, 0.75)
                
                # PLASTIC detection improvements
                is_plastic = False
                if ((saturation > 0.35 or brightness > 0.7) and 
                    gray_std < 35 and edge_density < 0.15 and
                    homogeneity > 0.7):
                    # Uniform texture, higher saturation or brightness,
                    # Low texture variation typical for plastic
                    is_plastic = True
                    if top1_idx != 4 or top1_val < 0.7:  # 4 is plastic class
                        top1_idx = 4
                        top1_val = max(top1_val, 0.75)
                
                # PAPER detection improvements
                is_paper = False
                if (saturation < 0.25 and gray_std < 25 and 
                    edge_density > 0.05 and energy > 0.2 and
                    gradient_std < 20):
                    # Low saturation, low texture variation,
                    # Medium edge density typical for paper
                    is_paper = True
                    if top1_idx != 3 or top1_val < 0.7:  # 3 is paper class
                        top1_idx = 3
                        top1_val = max(top1_val, 0.75)
                
                # CERAMIC detection (maps to glass for recyclability)
                is_ceramic = False
                if (gray_mean > 150 and saturation < 0.3 and
                    contrast < 0.3 and homogeneity > 0.8):
                    # Bright, low saturation, low contrast,
                    # High homogeneity typical for ceramics
                    is_ceramic = True
                    # In our class system, ceramics are not recyclable
                    top1_idx = 2  # Other/non-recyclable
                    top1_val = max(top1_val, 0.75)
                
                # Save these flags for the debug mode
                material_flags = {
                    "is_glass": is_glass,
                    "is_metal": is_metal,
                    "is_plastic": is_plastic,
                    "is_paper": is_paper,
                    "is_ceramic": is_ceramic
                }
                
                # Add texture analysis features to confidences 
                # for reporting in debug mode
                material_confidences["texture_features"] = {
                    "gray_std": float(gray_std),
                    "saturation": float(saturation),
                    "brightness": float(brightness),
                    "edge_density": float(edge_density),
                    "gradient_mean": float(gradient_mean),
                    "energy": float(energy),
                    "contrast": float(contrast),
                    "homogeneity": float(homogeneity),
                    "material_flags": material_flags
                }
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Texture analysis error: {e}")
                # If texture analysis fails, continue with original predictions
                material_confidences["texture_features"] = {"error": str(e)}
        
        # Use the final predictions
        class_idx = top1_idx
        confidence = top1_val

        # Determine if recyclable based on category
        is_recyclable = self.recyclable_categories.get(class_idx, False)
        material_type = self.class_names[class_idx]

        return {
            'is_recyclable': is_recyclable,
            'material_type': material_type,
            'class_idx': class_idx,
            'confidence': confidence,
            'material_confidences': material_confidences  # Add all material confidences for debug
        }

    def detect_product_category(self, class_name, confidence_threshold=0.6):
        """
        Detect product category based on class name and keywords.
        Returns product category and explanation if found, otherwise None.
        """
        # First check if it's a known item in our trash_items list
        if class_name in self.trash_items:
            category = self.trash_items[class_name]
            explanation = self.non_recyclable_categories.get(category, f"This is a {category} item")
            return category, explanation
            
        # Next, check if the class name contains any of our keywords
        for keyword, category in self.product_keywords.items():
            if keyword in class_name.lower():
                explanation = self.non_recyclable_categories.get(category, f"This is a {category} item")
                return category, explanation
                
        # If no match found, return None
        return None, None

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
            
            # Get the class name for this detection
            class_name = detections.names[int(class_id)].lower()
            
            # Detect product category based on class name first
            product_category, category_explanation = self.detect_product_category(class_name)
            
            # Special handling for eyeglasses which can be either plastic or metal
            is_eyeglasses = False
            if "glasses" in class_name or "eyeglasses" in class_name or "spectacles" in class_name:
                is_eyeglasses = True
                # We'll add it to potential waste for processing
                is_waste_object = True
            else:
                # Regular detection checks
                is_waste_object = (class_name in self.potential_waste_objects or 
                                class_name in self.trash_items or
                                class_name in self.food_waste or
                                any(waste in class_name for waste in ['trash', 'waste', 'recycle', 'garbage']))
            
            # Skip classification for objects that are definitely not waste
            if not is_waste_object:
                # Draw a gray bounding box for non-waste objects
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                label = f"{class_name}: Not Waste"
                
                # Draw label background and text
                font_scale = 1.0
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(annotated_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (128, 128, 128), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                continue
            
            # Crop the detected object for material classification
            object_crop = frame[y1:y2, x1:x2]
            
            # Skip if crop is empty
            if object_crop.size == 0:
                continue
            
            # Classify the material visually
            classification = self.classify_object(object_crop)
            material_type = classification['material_type']
            is_recyclable = classification['is_recyclable']
            confidence = classification['confidence']
            material_confidences = classification['material_confidences']
            
            # Special checks for specific item categories
            is_food_waste = class_name in self.food_waste
            
            # Apply special handling logic for different product types
            
            # Food waste handling
            if is_food_waste:
                material_type = 'food waste'
                is_recyclable = False
                waste_category = 'compostable'
                confidence = 0.9
                reason = "Food waste should be composted, not recycled"
            
            # Special handling for eyeglasses
            elif is_eyeglasses:
                # Check if eyeglasses are plastic or metal based on texture analysis
                if "texture_features" in material_confidences and "material_flags" in material_confidences["texture_features"]:
                    flags = material_confidences["texture_features"]["material_flags"]
                    
                    if flags.get("is_plastic", False):
                        material_type = "plastic"
                        is_recyclable = True
                        confidence = 0.85
                    elif flags.get("is_metal", False):
                        material_type = "metal"
                        is_recyclable = True
                        confidence = 0.85
                    else:
                        # Default to plastic if texture analysis is inconclusive
                        material_type = "plastic"
                        is_recyclable = True
                        confidence = 0.7
            
            # Apply product category logic if applicable
            elif product_category:
                # For e-waste, hazardous, specialized, etc. we override recyclability
                if product_category in ['e-waste', 'hazardous', 'specialized', 'mixed', 'non-recyclable']:
                    is_recyclable = False
                    # Keep the material type but indicate the product category
                    material_type = f"{material_type} ({product_category})"
                    reason = category_explanation
                
                # For items like plastic film that might be recyclable in special programs
                elif product_category == 'plastic-film' and material_type == 'plastic':
                    is_recyclable = False  # Not in regular recycling
                    reason = "Plastic film requires special recycling programs, not standard recycling bins"
                
                # For fabric items
                elif product_category == 'fabric':
                    is_recyclable = False
                    reason = "Textiles should be donated or taken to textile recycling centers"
            
            # Only use predefined mappings if no product category was found and confidence is low
            elif confidence < 0.6:
                # Use known mappings for certain categories as fallback
                if is_food_waste:
                    material_type = 'food waste'
                    is_recyclable = False
                    confidence = 0.9
                elif product_category:
                    # For e-waste, hazardous, specialized, etc. we override recyclability
                    if product_category in ['e-waste', 'hazardous', 'specialized', 'mixed', 'non-recyclable']:
                        is_recyclable = False
                        # Keep the material type but indicate the product category
                        material_type = f"{material_type} ({product_category})"
                        reason = category_explanation
                
                # For items like plastic film that might be recyclable in special programs
                elif product_category == 'plastic-film' and material_type == 'plastic':
                    is_recyclable = False  # Not in regular recycling
                    reason = "Plastic film requires special recycling programs, not standard recycling bins"
                
                # For fabric items
                elif product_category == 'fabric':
                    is_recyclable = False
                    reason = "Textiles should be donated or recycled through special textile programs"
            
            # Update classification with product context information
            classification = {
                'is_recyclable': is_recyclable,
                'material_type': material_type,
                'class_idx': classification['class_idx'],
                'confidence': confidence
            }
            
            if product_category:
                classification['product_category'] = product_category
                classification['reason'] = category_explanation
            elif is_food_waste:
                classification['reason'] = "Food waste should be composted, not recycled"
            
            # Determine waste category based on classification
            if material_type == 'food waste' or "food waste" in material_type:
                waste_category = 'compostable'
                color = (0, 128, 128)  # Brown color for food waste
                waste_label = "Compostable"
            elif is_recyclable:
                waste_category = 'recyclable'
                color = (0, 255, 0)  # Green for recyclable
                waste_label = "Recyclable"
            elif product_category and product_category in self.disposal_instructions:
                # Use the product category to determine waste handling
                waste_category = product_category
                # Specialized colors for different disposal methods
                if product_category == 'e-waste':
                    color = (255, 0, 255)  # Magenta for e-waste
                elif product_category == 'hazardous':
                    color = (0, 0, 128)  # Dark red for hazardous
                elif product_category == 'specialized':
                    color = (255, 128, 0)  # Orange for specialized recycling
                elif product_category == 'fabric':
                    color = (255, 0, 128)  # Pink for fabric
                elif product_category == 'plastic-film':
                    color = (128, 255, 255)  # Light blue for plastic film
                else:
                    color = (0, 0, 255)  # Default red for non-recyclable
                waste_label = product_category.title()
            else:
                waste_category = 'trash'
                color = (0, 0, 255)  # Red for non-recyclable trash
                waste_label = "Trash"

            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Add disposal instructions
            disposal_instructions = self.disposal_instructions.get(waste_category, "")
            
            # Prepare label text
            if 'reason' in classification:
                label = f"{class_name}: {material_type} ({waste_label})"
                reason = classification['reason']
            else:
                label = f"{class_name}: {material_type} ({waste_label})"
                reason = ""

            # Font settings
            font_scale = 0.8
            thickness = 2

            # Calculate total label height needed
            label_lines = [label]
            if reason:
                label_lines.append(reason)
            if disposal_instructions:
                label_lines.append(disposal_instructions)
                
            # Calculate total text height
            total_text_height = 0
            text_widths = []
            for line in label_lines:
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                total_text_height += text_height + 5  # Add padding
                text_widths.append(text_width)
                
            max_width = max(text_widths)

            # Draw label background - adjust size based on text
            bg_y1 = y1 - total_text_height - 10
            cv2.rectangle(annotated_frame, (x1, bg_y1), (x1 + max_width + 10, y1), color, -1)

            # Draw label text for each line
            current_y = y1 - 5
            for line in reversed(label_lines):  # Start from bottom line and move upwards
                cv2.putText(annotated_frame, line, (x1 + 5, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                (_, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                current_y -= (text_height + 5)  # Move up for next line

            # If debug mode is on, show all material confidence scores
            if self.debug_mode:
                # Print material confidences to console
                debug_str = " | ".join([f"{mat}: {conf:.2f}" for mat, conf in material_confidences.items() 
                                        if not isinstance(conf, dict)])
                print(f"Object: {class_name} | Materials: {debug_str}")
                
                # Print texture features if available
                if "texture_features" in material_confidences:
                    features = material_confidences["texture_features"]
                    if "error" not in features:
                        print(f"  Texture features:")
                        for k, v in features.items():
                            if k != "material_flags":
                                print(f"    {k}: {v:.3f}")
                        
                        # Print material flags
                        if "material_flags" in features:
                            flags = features["material_flags"]
                            flags_str = " | ".join([f"{k}: {v}" for k, v in flags.items() if v])
                            if flags_str:
                                print(f"  Material flags: {flags_str}")
                
                # Create a larger crop area to show debug info on frame
                y_text = y1 - 20
                # Show material confidences
                for mat, conf in material_confidences.items():
                    if not isinstance(conf, dict):
                        y_text -= 20
                        debug_text = f"{mat}: {conf:.2f}"
                        cv2.putText(annotated_frame, debug_text, (x1, y_text), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                # Show texture-based material detection results
                if "texture_features" in material_confidences and "material_flags" in material_confidences["texture_features"]:
                    flags = material_confidences["texture_features"]["material_flags"]
                    for flag_name, flag_value in flags.items():
                        if flag_value:
                            y_text -= 20
                            flag_text = f"{flag_name}: YES"
                            cv2.putText(annotated_frame, flag_text, (x1, y_text),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # After processing all detections, write to CSV if enabled
            if self.csv_writer and len(detections) > 0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for i, (bbox, cls_name, conf, material_type, is_recyclable, category, _) in enumerate(detections):
                    confidence_val = float(conf) if i < len(conf) else 0.0
                    self.csv_writer.writerow([
                        timestamp, 
                        cls_name, 
                        material_type, 
                        confidence_val, 
                        category,
                        "Yes" if is_recyclable else "No"
                    ])

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

    def __del__(self):
        if self.csv_file:
            self.csv_file.close()

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
    
    # Input source options
    input_group = parser.add_argument_group('Input options')
    input_group.add_argument('--webcam', action='store_true',
                       help='Use webcam as input source (default)')
    input_group.add_argument('--video', type=str, 
                       help='Path to video file for detection')
    input_group.add_argument('--image', type=str,
                       help='Path to image file for detection')
    
    # Output options
    parser.add_argument('--output', default=None,
                        help='Path to save output video (None for no output)')
    parser.add_argument('--detector', default='yolov8n.pt',
                        help='Path to YOLOv8 detector model')
    parser.add_argument('--classifier', default='recyclable_classifier.pt',
                        help='Path to trained classifier weights')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Confidence threshold for detection (0-1)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to show material classification scores')
    parser.add_argument('--csv', default=None,
                        help='Path to save detection results in CSV format')
    
    # Training options
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
            device=args.device,
            conf_threshold=args.conf_threshold,
            debug_mode=args.debug,
            csv_output=args.csv
        )

        # Determine input source
        if args.video:
            source = args.video
        elif args.image:
            # For image input, we'll process a single frame
            image = cv2.imread(args.image)
            if image is None:
                print(f"Error: Could not read image file {args.image}")
                return
            
            # Process the image
            processed_image = system.process_frame(image)
            
            # Display the result
            cv2.imshow("Recyclable Detection", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save output if specified
            if args.output:
                cv2.imwrite(args.output, processed_image)
                print(f"Output saved to {args.output}")
            
            return
        else:
            # Default to webcam (index 0)
            source = 0
            print("Using webcam as input source.")

        # Run detection on video source
        system.run_on_video(
            source=source,
            output=args.output,
            show=True
        )

if __name__ == "__main__":
    main()
