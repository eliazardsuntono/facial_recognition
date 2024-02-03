import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the webcam
vid = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not vid.isOpened():
    print('Error: Unable to open the webcam.')
    exit()

# Runs the camera loop
while True:
    # Read the frame from the webcam
    ret, frame = vid.read()

    # Check if the frame is read successfully
    if not ret:
        print('Error: Unable to read frame from the webcam.')
        break
    
    # Grey scales the color
    grey_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(grey_scale, 1.1, 9)

    # Draw rectangles around the faces and add text
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release the video capture object and close all windows
vid.release()
cv2.destroyAllWindows()
