import cv2

# Try to open the camera
print("Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully")

try:
    # Read a few frames
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        else:
            print(f"Frame {i} grabbed successfully, shape: {frame.shape}")
        
        # Display the frame
        cv2.imshow('Test', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error during camera operation: {e}")
finally:
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released")