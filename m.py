import streamlit as st

def main():
    st.title("JavaScript Camera in Streamlit")

    # Inject the JavaScript code
    js_code = """
    <script>
        // Function to capture and display the camera stream
        function startCamera() {
            const video = document.getElementById("video");

            // Access the camera stream
            navigator.mediaDevices.getUserMedia({ video: true })
                .then((stream) => {
                    video.srcObject = stream;
                })
                .catch((error) => {
                    console.error("Error accessing camera: ", error);
                });
        }

        // Function to capture and display a photo
        function capturePhoto() {
            const video = document.getElementById("video");
            const canvas = document.getElementById("canvas");
            const context = canvas.getContext("2d");

            // Draw the current video frame onto the canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert the canvas image to base64 data URL
            const photoDataUrl = canvas.toDataURL("image/png");

            // Display the captured photo
            const photoElement = document.getElementById("photo");
            photoElement.src = photoDataUrl;
import streamlit as st
import base64

def main():
    st.title("JavaScript Camera in Streamlit")

    # Inject the JavaScript code
    js_code = """
    <script>
        // Function to capture and display the camera stream
        function startCamera() {
            const video = document.getElementById("video");

            // Access the camera stream
            navigator.mediaDevices.getUserMedia({ video: true })
                .then((stream) => {
                    video.srcObject = stream;
                })
                .catch((error) => {
                    console.error("Error accessing camera: ", error);
                });
        }

        // Function to capture and display a photo
        function capturePhoto() {
            const video = document.getElementById("video");
            const canvas = document.getElementById("canvas");
            const context = canvas.getContext("2d");

            // Draw the current video frame onto the canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert the canvas image to base64 data URL
            const photoDataUrl = canvas.toDataURL("image/png");

            // Send the captured photo data URL to the Streamlit app
            const resultElement = document.getElementById("result");
            resultElement.value = photoDataUrl;
        }
    </script>
    """

    # Display the camera and photo capture UI
    st.markdown(js_code, unsafe_allow_html=True)
    st.write('<button onclick="startCamera()">Start Camera</button>', unsafe_allow_html=True)
    st.write('<button onclick="capturePhoto()">Capture Photo</button>', unsafe_allow_html=True)
    result_element = st.empty()  # Create an empty placeholder for the photo

    # Create an HTML canvas element
    st.write('<canvas id="canvas" width="640" height="480" style="display: none;"></canvas>', unsafe_allow_html=True)

    # Create an HTML video element
    st.write('<video id="video" width="640" height="480" autoplay playsinline style="border: 1px solid black;"></video>', unsafe_allow_html=True)

    # Retrieve the captured photo from the JavaScript code
    photo_data_url = st._get_forward_msg("result")["content"]["data"]["text/plain"]

    # Display the captured photo
    if photo_data_url:
        # Convert the data URL to base64
        _, encoded = photo_data_url.split(",", 1)
        decoded = base64.b64decode(encoded)
        st.image(decoded, use_column_width=True)

if __name__ == "__main__":
    main()


# Add any other Streamlit components or functionality as needed

# import streamlit as st

# picture = st.camera_input("Take a picture")
# if picture:
#     st.image(picture)
#     with open("esong", "wb") as f:
#         f.write(picture)

# import cv2
# import argparse
# import numpy as np
# from keras.models import model_from_json
# from keras.models import model_from_json
# from keras.utils import load_img, img_to_array
# from PIL import Image
# import streamlit as st
# # Parse the arguments
# json_file = open('fer.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)

# # Load weights and them to model
# model.load_weights('fer.h5')

# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# picture = st.camera_input("Take a picture")
# if picture:
#     st.image(picture)
#     # Convert the picture to a NumPy array and read it using OpenCV
#     picture_bytes = picture.read()
#     # Convert the picture to a NumPy array and read it using OpenCV
#     nparr = np.fromstring(picture_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces_detected = classifier.detectMultiScale(gray_img, 1.18, 5)

#     for (x, y, w, h) in faces_detected:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         roi_gray = gray_img[y:y + w, x:x + h]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         img_pixels = img_to_array(roi_gray)
#         img_pixels = np.expand_dims(img_pixels, axis=0)
#         img_pixels /= 255.0

#         predictions = model.predict(img_pixels)
#         max_index = int(np.argmax(predictions))

#         emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
#         predicted_emotion = emotions[max_index]

#         cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

#     resized_img = cv2.resize(img, (1024, 768))
#     cv2.imshow('Facial Emotion Recognition', resized_img)

#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()