import cv2

vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml') 


# Runs the camera loop
while True:
    ret, frame = vid.read()
    
    # Grey scales the color
    grey_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_scale, 1.1, 9)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
vid.release()
cv2.destroyAllWindows()