# from PIL import Image
# import streamlit as st
# import time
# st.title("Guren")      

# def fun():
#         mood = None

#         # Create two columns
#         col1, col2 = st.columns(2)

#         # Set the image size
#         image_size = 100

#         # Load and resize the first image
#         image1 = Image.open("happy.png")
#         image1 = image1.resize((image_size, image_size))

#         # Display the first image in the first column
#         col1.image(image1)
#         if col1.button("Happy"):
#             mood = "happy"

#         # Load and resize the second image
#         image2 = Image.open("neutral.jpg")
#         image2 = image2.resize((image_size, image_size))

#         # Display the second image in the first column
#         col1.image(image2)
#         if col1.button("Neutral"):
#             mood = "neutral"

#         # Load and resize the third image
#         image3 = Image.open("sad.jpg")
#         image3 = image3.resize((image_size, image_size))

#         # Display the third image in the second column
#         col2.image(image3)
#         if col2.button("Sad"):
#             mood = 'sadness'

#         # Load and resize the fourth image
#         image4 = Image.open("angry.jpg")
#         image4 = image4.resize((image_size, image_size))

#         # Display the fourth image in the second column
#         col2.image(image4)
#         if col2.button("Angry"):
#             mood = "angry"

#         # Display the selected mood
        
#         print("modd ",mood)
#         if mood:
#             st.write(f"You selected: {mood}")
            
#         return mood

# m = None
# while not m:
#     m = fun()
#     time.sleep(0.1)
# print("m ",m)
import streamlit as st
from PIL import Image
def my_function():
        mood = None

        # Create two columns
        col1, col2 = st.columns(2)

        # Set the image size
        image_size = 100

        # Load and resize the first image
        image1 = Image.open("happy.png")
        image1 = image1.resize((image_size, image_size))

        # Display the first image in the first column
        col1.image(image1)
        if col1.button("Happy"):
            mood = "happy"

        # Load and resize the second image
        image2 = Image.open("neutral.jpg")
        image2 = image2.resize((image_size, image_size))

        # Display the second image in the first column
        col1.image(image2)
        if col1.button("Neutral"):
            mood = "neutral"

        # Load and resize the third image
        image3 = Image.open("sad.jpg")
        image3 = image3.resize((image_size, image_size))

        # Display the third image in the second column
        col2.image(image3)
        if col2.button("Sad"):
            mood = 'sadness'

        # Load and resize the fourth image
        image4 = Image.open("angry.jpg")
        image4 = image4.resize((image_size, image_size))

        # Display the fourth image in the second column
        col2.image(image4)
        if col2.button("Angry"):
            mood = "angry"

        # Display the selected mood
        
        print("modd ",mood)
        if mood:
            st.write(f"You selected: {mood}")
            
        return mood

choice = my_function()

if choice:
    st.write(f'You chose: {choice}')
    print("lllll",choice)