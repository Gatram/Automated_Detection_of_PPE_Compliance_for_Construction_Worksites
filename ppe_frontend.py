from flask import Flask, Response, render_template, jsonify
import cv2
import time
from threading import Thread
from ultralytics import YOLO

app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO("models/best_030124.pt")  # Update with your model path

# Initialize webcam
cap = cv2.VideoCapture("C:/Users/gatra/Downloads/PPE-detection-of-sample-in-test-photo-Figure-7-demonstrates-the-PPE-detection-models.png")  # 0 for the default webcam

# Variable to store the latest message and timestamp
latest_message = ""
message_timestamp = 0

@app.route('/')
def index():
    # Video streaming home page
    return render_template('index.html')

def generate_frames():
    global latest_message, message_timestamp
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)  # Inference

        # Process the results
        annotated_frame = results[0].plot()  # Draw bounding boxes on the frame
        detected_objects = [model.names[int(box.cls)] for box in results[0].boxes]
        class_name = ""
        for i in detected_objects:
            if i == "no glove" or i == "no helmet" or i == "no vest" or i == "no boots":
                i = i[3:]
                class_name += i + ","
        class_name = class_name[:-1]

        # Update the message every 10 seconds
        current_time = time.time()
        if class_name and (current_time - message_timestamp > 3):
            latest_message = f"Please wear the {class_name}"
            message_timestamp = current_time

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/get_message')
def get_message():
    return jsonify({"message": latest_message})

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
