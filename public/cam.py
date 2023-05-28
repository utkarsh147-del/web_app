import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils import load_img, img_to_array
from keras.models import load_model
import time
# Load model from JSON file
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load pre-trained emotion detection model
model = model.load_weights('fer.h5')

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize camera feed
camera = cv2.VideoCapture(0)

# Continuously capture frames from camera feed
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    
    # Convert frame to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    
    # Reshape image to match model input shape
    reshaped = resized.reshape(1, 48, 48, 1)
    
    # Normalize image pixel values
    normalized = reshaped / 255.0
    
    # Predict emotion from image
    prediction = model.predict(normalized)
    
    # Get emotion label with highest probability
    emotion_label = emotions[np.argmax(prediction)]
    
    # Draw emotion label on frame
    cv2.putText(frame, emotion_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Exit program when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close window
camera.release()
cv2.destroyAllWindows()
