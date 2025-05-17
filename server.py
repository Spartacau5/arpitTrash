import os
import sys
import cv2
import time
import base64
import threading
from flask import Flask, render_template, Response, jsonify
from recyclable_detection import RecyclableDetectionSystem

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
detection_system = None
camera_running = False

def initialize_system(detector_model='yolov8n.pt', classifier_weights='recyclable_classifier.pt', debug_mode=False):
    global detection_system
    detection_system = RecyclableDetectionSystem(
        detector_model=detector_model,
        classifier_weights=classifier_weights,
        debug_mode=debug_mode
    )
    return detection_system

def get_camera(source=0):
    global camera
    if camera is None:
        print(f"Attempting to initialize camera with source {source}")
        camera = cv2.VideoCapture(source)
        if not camera.isOpened():
            print(f"Error: Could not open video source {source}")
            # Try alternative source if default fails
            camera = cv2.VideoCapture(1)
            if not camera.isOpened():
                print("Error: Could not open any video source")
                return None
        print("Camera initialized successfully")
        # Set some basic properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Small delay to ensure camera is ready
        time.sleep(1)
    return camera

def process_frames():
    global camera, output_frame, lock, detection_system, camera_running
    
    camera_running = True
    fps_count = 0
    fps = 0
    fps_start_time = time.time()
    
    while camera_running:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            break
            
        start_time = time.time()
        
        # Process the frame with our detection system
        if detection_system:
            processed_frame = detection_system.process_frame(frame)
            
            # Add FPS counter
            fps_count += 1
            if (time.time() - fps_start_time) > 1.0:  # Update FPS every second
                fps = fps_count / (time.time() - fps_start_time)
                fps_count = 0
                fps_start_time = time.time()
                
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            processed_frame = frame
            
        # Acquire lock and update output frame
        with lock:
            output_frame = processed_frame.copy()
            
        # Sleep to maintain reasonable FPS
        time.sleep(max(0.01, 0.03 - (time.time() - start_time)))
    
    # Release camera when done
    if camera is not None:
        camera.release()

def generate_frames():
    global output_frame, lock
    
    while True:
        # Wait until we have a frame
        with lock:
            if output_frame is None:
                continue
            frame = output_frame.copy()
            
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        # Convert to bytes
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Home page with video stream."""
    # Render the index template
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for streaming video."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """Start the camera and processing thread."""
    global camera, camera_running
    
    if camera_running:
        return jsonify({"status": "Camera already running"})
        
    try:
        # Initialize camera
        print("Starting camera initialization...")
        cam = get_camera(0)  # Use webcam as default
        
        if cam is None:
            return jsonify({"status": "Error", "message": "Could not initialize camera"})
            
        # Test if we can read a frame
        success, frame = cam.read()
        if not success:
            return jsonify({"status": "Error", "message": "Could not read from camera"})
            
        print("Camera initialized and tested successfully")
        
        # Start processing thread
        threading.Thread(target=process_frames, daemon=True).start()
        
        return jsonify({"status": "Camera started successfully"})
    except Exception as e:
        print(f"Error starting camera: {str(e)}")
        return jsonify({"status": "Error", "message": str(e)})

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera and processing thread."""
    global camera_running
    
    camera_running = False
    return jsonify({"status": "Camera stopped"})

def create_templates():
    """Create the templates directory and HTML template"""
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recyclable Object Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                text-align: center;
            }
            h1 {
                color: #388e3c;
            }
            .video-container {
                margin: 20px 0;
                padding: 10px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            #video-stream {
                max-width: 100%;
                height: auto;
            }
            .controls {
                margin-top: 20px;
            }
            button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
            }
            button:hover {
                background-color: #388e3c;
            }
            #stop-btn {
                background-color: #f44336;
            }
            #stop-btn:hover {
                background-color: #d32f2f;
            }
            .legend {
                margin-top: 20px;
                text-align: left;
                display: inline-block;
                background-color: #fff;
                padding: 10px 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .legend-item {
                margin: 10px 0;
                display: flex;
                align-items: center;
            }
            .color-box {
                width: 20px;
                height: 20px;
                margin-right: 10px;
                display: inline-block;
            }
            .recyclable {
                background-color: #4CAF50;
            }
            .non-recyclable {
                background-color: #f44336;
            }
            .compostable {
                background-color: #808000;
            }
            .e-waste {
                background-color: #ff00ff;
            }
            .hazardous {
                background-color: #800000;
            }
            .specialized {
                background-color: #ff8000;
            }
            .not-waste {
                background-color: #808080;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Recyclable Object Detection System</h1>
            
            <div class="video-container">
                <img id="video-stream" src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
            
            <div class="controls">
                <button id="start-btn" onclick="startCamera()">Start Camera</button>
                <button id="stop-btn" onclick="stopCamera()">Stop Camera</button>
            </div>
            
            <div class="legend">
                <h3>Classification Legend:</h3>
                <div class="legend-item">
                    <div class="color-box recyclable"></div>
                    <span>Recyclable - Place in regular recycling bin</span>
                </div>
                <div class="legend-item">
                    <div class="color-box non-recyclable"></div>
                    <span>Non-Recyclable - Place in regular trash bin</span>
                </div>
                <div class="legend-item">
                    <div class="color-box compostable"></div>
                    <span>Compostable - Place in compost bin</span>
                </div>
                <div class="legend-item">
                    <div class="color-box e-waste"></div>
                    <span>E-Waste - Take to e-waste collection center</span>
                </div>
                <div class="legend-item">
                    <div class="color-box hazardous"></div>
                    <span>Hazardous - Take to hazardous waste facility</span>
                </div>
                <div class="legend-item">
                    <div class="color-box specialized"></div>
                    <span>Specialized Recycling - Requires special recycling program</span>
                </div>
                <div class="legend-item">
                    <div class="color-box not-waste"></div>
                    <span>Not Waste - Regular object, not considered waste</span>
                </div>
            </div>
        </div>

        <script>
            function startCamera() {
                fetch('/start_camera')
                    .then(response => response.json())
                    .then(data => console.log(data));
            }
            
            function stopCamera() {
                fetch('/stop_camera')
                    .then(response => response.json())
                    .then(data => console.log(data));
            }
            
            // Start camera automatically when page loads
            document.addEventListener('DOMContentLoaded', function() {
                startCamera();
            });
        </script>
    </body>
    </html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(index_html)

if __name__ == '__main__':
    # Create template directory and files
    create_templates()
    
    # Initialize the detection system
    initialize_system(debug_mode=False)
    
    # Run the Flask app
    print("Starting Recyclable Detection Web Server...")
    print("Open your browser and navigate to http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False) 