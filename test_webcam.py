import cv2

def test_camera():
    print("Attempting to access camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera 0, trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Could not open any camera")
            return False
    
    print("Successfully opened camera")
    print("Reading test frame...")
    ret, frame = cap.read()
    if ret:
        print("Successfully read a frame")
        cv2.imwrite('test_frame.jpg', frame)
        print("Saved test frame as test_frame.jpg")
    else:
        print("Failed to read frame")
    
    cap.release()
    return ret

if __name__ == "__main__":
    test_camera() 