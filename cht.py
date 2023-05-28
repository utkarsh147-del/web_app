import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained Haar cascades for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained model for facial expression recognition
model = load_model('emotion_detection_model.h5')

# Define a dictionary to map class labels to human-readable emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face ROI to 48x48 pixels (required by the model)
        face_roi = cv2.resize(face_roi, (48, 48))

        # Normalize the pixel values (required by the model)
        face_roi = face_roi / 255.0

        # Convert the face ROI to a 4D tensor (required by the model)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Predict the emotion label for the face ROI using the model
        pred = model.predict(face_roi)[0]
        emotion_label = emotion_labels[np.argmax(pred)]

        # Draw a rectangle around the face and show the predicted emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the window
cap.release()
cv2.destroyAllWindows()
