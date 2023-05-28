# import streamlit as st
# import cv2
# import numpy as np
# from keras.models import model_from_json
# from keras.utils import load_img, img_to_array
# from tensorflow.keras.preprocessing.image import img_to_array
# import time
# from streamlit_webrtc import st_webrtc


# json_file = open('fer.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)

# # Load weights and them to model
# model.load_weights('fer.h5')

# face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Define the WebRTC video feed
# RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# video_feed = st_webrtc.VideoTransformer(
#     src_size=(640, 480), rtc_configuration=RTC_CONFIGURATION, key="emotion-detection"
# )

# # Define the emotions
# emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# def predict_emotion(img):
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

#     for (x, y, w, h) in faces_detected:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
#         roi_gray = gray_img[y:y + w, x:x + h]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         img_pixels = img_to_array(roi_gray)
#         img_pixels = np.expand_dims(img_pixels, axis=0)
#         img_pixels /= 255.0

#         predictions = model.predict(img_pixels)
#         max_index = int(np.argmax(predictions))
#         predicted_emotion = emotions[max_index]

#         cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

#     return img

# def transform(image):
#     img = image.to_ndarray(format="bgr24")
#     img = predict_emotion(img)
#     return img

# # Initialize the WebRTC video feed
# video_feed = st_webrtc.VideoTransformer(
#     src_size=(640, 480), 
#     transform_func=transform, 
#     rtc_configuration=RTC_CONFIGURATION, 
#     key="emotion-detection"
# )

# start_time = time.time()
# while (time.time() - start_time) <= 50:
#     # Display the video feed in the Streamlit app
#     video_display = video_feed()

#     # Display the predicted emotion on the video feed
#     st.image(video_display, caption="Predicted Emotion")
    
#     # Display the predicted emotion after 20 seconds
#     if (time.time() - start_time) > 20:
#         st.write("Your emotion:", predicted_emotion)


import cv2
import streamlit as st
import av
import numpy as np
from keras.models import model_from_json
from keras.utils import load_img, img_to_array
import time
# Load model from JSON file
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights and them to model
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define Streamlit WebRTC video configuration
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Define function to process video frames
@st.cache(allow_output_mutation=True)
def process_frame(frame):
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

        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return img

# Define Streamlit app
def app():
    st.title("Facial Emotion Recognition")

    # Define WebRTC video player
    webrtc_ctx = st.webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=process_frame,
        async_processing=True,
    )

    # Display predicted emotion after 20 seconds of video processing
    if webrtc_ctx.video_receiver and time.time() - webrtc_ctx.start_time >= 20:
        predicted_emotion = ...
        st.write("Your emotion:", predicted_emotion)

if __name__ == "__main__":
    app()
