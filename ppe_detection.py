import cv2

from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("models/best_030124.pt")  # Update with your model path

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam, change if you have multiple cameras

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
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
            class_name += i+","
    class_name = class_name[:-1]
    print(f"please wear the {class_name}")
    # Display the resulting frame
    cv2.imshow('YOLOv8 Real-Time Inference', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118