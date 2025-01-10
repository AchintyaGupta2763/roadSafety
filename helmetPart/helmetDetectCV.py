import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("helmetPart/runs/detect/train/weights/best.pt")

# Open the video file using OpenCV
video_path = 'helmetPart/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define colors for each label
colors = {
    "with helmet": (0, 255, 0),    # Green for "with helmet"
    "without helmet": (0, 0, 255)  # Red for "without helmet"
}

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use the YOLO model to make predictions on the current frame
    results = model.predict(source=frame)

    # Iterate over each detection result
    for result in results:
        boxes = result.boxes.cpu().numpy()  # Get bounding boxes
        labels = result.names  # Get class names
        
        # Draw bounding boxes and labels on the frame
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            label = labels[int(box.cls)]  # Label for the detected class
            confidence = box.conf[0]  # Confidence score

            # Choose color based on the label
            color = colors.get(label, (255, 255, 255))  # Default to white if label is unknown

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Display the label and confidence score
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the annotated frame using OpenCV
    cv2.imshow('Helmet Detection', frame)

    # Press 'q' to quit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
