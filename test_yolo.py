import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully")

# Open the webcam
print("Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

print("Camera opened successfully")

try:
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error during detection: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released")