import cv2

# Try different indices: 0, 1, 2, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open DroidCam (try a different index like 0 or 2)")
else:
    print("DroidCam opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    cv2.imshow('DroidCam USB Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
