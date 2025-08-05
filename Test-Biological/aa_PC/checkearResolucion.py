import cv2

cap = cv2.VideoCapture(0)

print(cap.isOpened())

cap.set(3,160)
cap.set(4,120)
ret, frame = cap.read()

cv2.imshow('display', frame)
print(frame.shape)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)