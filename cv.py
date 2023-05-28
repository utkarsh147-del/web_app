import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils import load_img, img_to_array
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import csv
# Load model from JSON file
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights and them to model
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
latest_emotion = []
# ite=9

                    
class VideoTransformer(VideoTransformerBase):
        latest_emotion = None
        def __init__(self):
            self.latest_emotions = []
            self.csv_file = open('emotions.csv', 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
        def transform(self, frame):
            self.csv_file.seek(0)
            self.csv_file.truncate()
            self.csv_writer.writerow(['Emotion'])
            global latest_emotion
            VideoTransformer.latest_emotion = latest_emotion
            img = frame.to_ndarray(format="bgr24")
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))
            for (x, y, w, h) in faces_detected:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255.0
                predictions = model.predict(img_pixels)
                max_index = int(np.argmax(predictions))
                emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
                predicted_emotion = emotions[max_index]
                # latest_emotion=predicted_emotion
                self.latest_emotions.append(predicted_emotion)
                self.csv_writer.writerow([predicted_emotion])
                    
                print(self.latest_emotions)
                st.success(latest_emotion)
                cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                # print(self.iter)
                # self.ite+=1 
            return img
        # def close(self):
        #     self.csv_file.close()

print("llll",latest_emotion)
def fun():
    st.title("Facial Emotion Recognition")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
    return latest_emotion
    
emotions = fun()


emotion = None

with open('emotions.csv', 'r', newline='') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    try:
        data_row = next(csv_reader)
        emotion = data_row['Emotion']
    except StopIteration:
        # The CSV file is empty, or there is no data in the first row
        pass

if emotion is not None:
    # Do something with the emotion
    st.write(f'Your mood is {emotion}')
else:
    # The CSV file is empty or there was no data in the first row
    print('No emotion data found in the CSV file')  
# st.write(f'Your mood is {csv_reader}')

