from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("rvm_best_yolov8s.pt")   # تأكد إن اسم الملف صح

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = laptop camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("RVM Live Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()