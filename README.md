
```bash
source venv/bin/activate
python recyclable_detection.py --mode detect
```
# Recyclable Object Detection System

A real-time computer vision system that detects objects in video streams and classifies them as recyclable or non-recyclable materials.

## Overview

This project uses:
- **YOLOv8** for object detection
- **Custom classifier** trained on the TrashNet dataset for recyclable material identification
- **Real-time processing** for webcam or video input

The system identifies common objects and classifies materials as:
- ♻️ **Recyclable**: glass, metal, paper, plastic
- ⚠️ **Non-recyclable**: other/trash

## Features

- Real-time object detection
- Material type classification
- Visual indicators (green for recyclable, red for non-recyclable)
- Performance metrics display (FPS counter)
- Webcam and video file support
- Training mode for custom classifiers

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- Ultralytics (YOLOv8)
- Pillow
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/recyclable-detection.git
   cd recyclable-detection
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision opencv-python ultralytics pillow numpy
   ```

3. Download the TrashNet dataset, or use your own dataset following the structure below:
   ```
   data/
     ├── cardboard/
     ├── glass/
     ├── metal/
     ├── paper/
     ├── plastic/
     └── trash/
   ```

## Dataset Preparation

Run the data organization script to prepare the dataset for training:

```bash
python organize_data.py
```

This script:
- Creates train/validation splits (80%/20%)
- Maps categories for training (cardboard → paper, trash → other)
- Saves the organized data in `data/trashnet-prepared/`

## Training the Classifier

Train the recyclable material classifier with:

```bash
python recyclable_detection.py --mode train --data_dir data/trashnet-prepared --epochs 5
```

Parameters:
- `--data_dir`: Directory containing the organized data
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--classifier`: Output path for the model (default: recyclable_classifier.pt)

The trained model will be saved as `recyclable_classifier.pt`.

## Running Object Detection

Run the detection system on your webcam:

```bash
python recyclable_detection.py --mode detect
```

Options:
- `--source`: Video source (0 for webcam, or path to video file)
- `--output`: Path to save output video
- `--detector`: Path to YOLOv8 model (default: yolov8n.pt)
- `--classifier`: Path to trained classifier (default: recyclable_classifier.pt)
- `--device`: Device to use (cpu, cuda, or None for auto-detect)

## How It Works

1. The system captures video frames from the webcam or video file
2. YOLOv8 detects objects in each frame
3. Detected objects are cropped and sent to the recyclable material classifier
4. Classification results determine recyclability status
5. The system displays bounding boxes with labels:
   - Green: Recyclable materials
   - Red: Non-recyclable materials

## Sample Output

The detection output includes:
- Object class (e.g., bottle, can)
- Material type (e.g., plastic, metal)
- Recyclability status
- Detection confidence

## Project Structure

- `recyclable_detection.py`: Main implementation
- `organize_data.py`: Dataset preparation script
- `data/`: Dataset directory
- `recyclable_classifier.pt`: Trained classifier model
- `yolov8n.pt`: YOLOv8 object detection model

## License

[MIT License](LICENSE)

## Acknowledgments

- TrashNet dataset: [https://github.com/garythung/trashnet](https://github.com/garythung/trashnet)
- YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)