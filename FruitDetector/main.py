from ultralytics import YOLO
import cv2

# load model
model = YOLO("best.pt")

# open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run detection
    results = model(frame, conf=0.5)

    # render bounding boxes
    annotated = results[0].plot()

    # show window
    cv2.imshow("Fruit Detector", annotated)

    # exit with ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
