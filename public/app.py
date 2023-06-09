import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils import load_img, img_to_array
import time
import streamlit as st
# Basic Streamlit Settings
st.set_page_config(page_title='eSangeet', layout = 'wide', initial_sidebar_state = 'auto')
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import subprocess
from load_css import local_css
from PIL import Image
import pydeck as pdk
import plotly.figure_factory as ff
import base64
import streamlit.components.v1 as components
import webbrowser
import random
#from streamlit import app_mode

# Loading css file
local_css("style.css")
if 'app_mode' not in st.session_state:
    st.session_state['app_mode'] = 'value'

# Session State also supports the attribute based syntax


# Sidebar Section
def spr_sidebar():
    with st.sidebar:
        # st.image(SPR_SPOTIFY_URL, width=60)
        st.info('**eSangeet**')
      #  home_button = st.button("About Me")
        data_button = st.button("Dataset")
        rec_button = st.button('Recommendation Engine')
        algo_button = st.button('Algorithm and Prediction')
      #  conc_button = st.button('Conclusion')
      #  report_button = st.button('My 4 weeks Progress Report')
      #  st.success('By Udit Katyal')
        st.session_state.log_holder = st.empty()
        # log_output('None')
      #  if home_button:
      #      st.session_state.app_mode = 'home'
        if data_button:
            st.session_state.app_mode = 'dataset'
        if algo_button:
            st.session_state.app_mode = 'algo'
        if rec_button:
            st.session_state.app_mode = 'recommend'
     #   if conc_button:
     #       st.session_state.app_mode = 'conclusions'
     #   if report_button:
     #       st.session_state.app_mode = 'report'

# Dataset Page
def dataset_page():
    st.markdown("<br>", unsafe_allow_html=True)
    """
    # Spotify Gen Track Dataset
    -----------------------------------
    Here I am using Spotity Gen Track Dataset, this dataset contains n number of songs and some metadata is included as well such as name of the playlist, duration, number of songs, number of artists, etc.
    """

    dataset_contains = Image.open('images/dataset_contains.png')
    st.image(dataset_contains, width =900)

    """
   
    - The data has three files namely spotify_albums, spotify_artists and spotify_tracks from which I am extracting songs information.
    - There is spotify features.csv files which contains all required features of the songs that I am using. 
    - The Spotify Song popularity based on location. 

    """
    """
    # Enhancing the data:
    These are some of the features that are available to us for each song and I am going to use them to enhance our dataset and to help matching
    the user's favorite song as per his/her input.

    ### These features value is measured mostly in a scale of 0-1:
    - **acousticness:** Confidence measure from 0.0 to 1.0 on if a track is acoustic.
    - **danceability:** Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo,
    rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
    - **energy:** Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically,
    energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale.
    Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
    - **instrumentalness:** Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or
    spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.
    Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
    - **liveness:** Detects the presence of an audience in the recording. Higher liveness values represent an increased probability
    that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
    - **loudness:** The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful
    for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical
    strength (amplitude). Values typical range between -60 and 0 db.
    - **speechiness:** Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording
    (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably
    made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in
    sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
    - **tempo:** The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the
    speed or pace of a given piece and derives directly from the average beat duration.
    - **valence:** A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound
    more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

    Refered docs: [link](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)
    """

    st.markdown("<br>", unsafe_allow_html=True)
    '''
    # Final Dataset 
    '''
    '''
    - Enhanced data
    '''
    dataframe1 = pd.read_csv('filtered_track_df1.csv')
    st.dataframe(dataframe1)
    st.markdown("<br>", unsafe_allow_html=True)


# Footer Section
def spr_footer():
    st.markdown('---')
    st.markdown(
        '© Copyright 2023 -  By Capstone team')


# 4_week_report_page
def report_page():
    st.markdown("<br>", unsafe_allow_html=True)

    st.header("Problems I ran through")
    '''
    - Data Collection -- Even though the core dataset I used was provided by Spotify, it was needed to go and look for other data sources to enhance the data and combine it with the core data set. 

    - Efficient Data Processing -- Initially I decided to take complete SpotifyGenTrack dataset (approximately 773.84 MB of data). But it became very important to enhance the data quality unless the recommendation might run to unnecessary bugs and outcomes. Therefore after many attempts, generated a filteredTrack csv which had the exact and refined dataset that was required for recommendation and predicting popularity of songs.

    - Unsupervised Learning -- Decided to take an different approach where I Explored different families of cluster algorithms and learning about advantages and disadvantages to make the best selection.

    '''
    st.text('If you are viewing on Hosted Url PDF wont be visible' )
    st.text('You can download it from here or from the github repo')

    with open("4_weeks_progress_report.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    

    st.download_button(label="Download Progress Report", 
        data=PDFbyte,
        file_name="ProgressReport.pdf",
        mime='application/octet-stream')
    def show_pdf(file_path):
        with open(file_path,"rb") as f:
          base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="900" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
    show_pdf("4_weeks_progress_report.pdf")

    
# Algorithm and Prediction Page 
def algo_page():
    st.header("1. Calculate Algorithms Accuracy")
    st.markdown(
        'Trainig the model and using Popularity as a Y-parameter to judge how accurate the algorithm comes out')

    st.header("Algorithms")
    st.subheader("Linear Regression")
    code = '''LR_Model = LogisticRegression()
    LR_Model.fit(X_train, y_train)
    LR_Predict = LR_Model.predict(X_valid)
    LR_Accuracy = accuracy_score(y_valid, LR_Predict)
    print("Accuracy: " + str(LR_Accuracy))

    LR_AUC = roc_auc_score(y_valid, LR_Predict)
    print("AUC: " + str(LR_AUC))

    Accuracy: 0.7497945543198379
    AUC: 0.5'''
    st.code(code, language='python')

   
    st.subheader("K-Nearest Neighbors Classifier")
    code = '''KNN_Model = KNeighborsClassifier()
    KNN_Model.fit(X_train, y_train)
    KNN_Predict = KNN_Model.predict(X_valid)
    KNN_Accuracy = accuracy_score(y_valid, KNN_Predict)
    print("Accuracy: " + str(KNN_Accuracy))

    KNN_AUC = roc_auc_score(y_valid, KNN_Predict)
    print("AUC: " + str(KNN_AUC))

    Accuracy: 0.7763381361967896
    AUC: 0.6890904291795135'''
    st.code(code, language='python')


    st.subheader("Decision Tree Classifier")
    code = '''DT_Model = DecisionTreeClassifier()
    DT_Model.fit(X_train, y_train)
    DT_Predict = DT_Model.predict(X_valid)
    DT_Accuracy = accuracy_score(y_valid, DT_Predict)
    print("Accuracy: " + str(DT_Accuracy))

    DT_AUC = roc_auc_score(y_valid, DT_Predict)
    print("AUC: " + str(DT_AUC))

    Accuracy: 0.8742672437407549
    AUC: 0.8573960839474465
    '''
    st.code(code, language='python')

    
    st.subheader("Random Forest")
    code = '''RFC_Model = RandomForestClassifier()
    RFC_Model.fit(X_train, y_train)
    RFC_Predict = RFC_Model.predict(X_valid)
    RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
    print("Accuracy: " + str(RFC_Accuracy))

    RFC_AUC = roc_auc_score(y_valid, RFC_Predict)
    print("AUC: " + str(RFC_AUC))

    Accuracy: 0.9357365912452748
    AUC: 0.879274665020435'''
    st.code(code, language='python')


    st.header("2. Popularity By Location")
    '''
    For Input purpose I took the most listened song from my dataset **Blinding Lights**  and predicted it's popularity score
    
    '''

    top_10_tracks = Image.open("images/top_tracks.png")
    st.image(top_10_tracks , caption ="Top 1o Tracks", width = 800)
    
    # 3-D EARTH MODEL
    st.header("3-D Earth Model")
    '''
    THREE GRAPH -:  [ 3-D EARTH MODEL ](http://threegraphs.com/charts/preview/9036/embed/)
    
    '''
    
    # url = 'http://threegraphs.com/charts/preview/9036/embed/'

    # if st.button('3-D Earth Model'):
    #     webbrowser.open_new_tab(url)
    
    st.text("To view the model on WebApp run the application on Local Host")
    
    st.code("Popularity ranges from 0 - 100 but to make visible on map 1 Unit = 1000 Unit, for instance 32200 score = 32.2 popularity")
    components.iframe("http://threegraphs.com/charts/preview/9036/embed/", width = 1000, height = 700)

   


# Load Data and n_neighbors_uri_audio are helper functions inside Recommendation Page
# Loads the track from filtered_track_df.csv file
def load_data():
    df = pd.read_csv(
        "filtered_track_df1.csv")
    df['genres'] = df.genres.apply(
        lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df


#genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
 #              'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
genre_names=['happiness', 'sadness', 'anger','neutral']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

#genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
 #              'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
genre_names=['happiness', 'sadness', 'anger','neutral']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

# Fetches the Nearest Song according to Genre start_year and end year.
def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"] == genre) & (
        exploded_track_df["release_year"] >= start_year) & (exploded_track_df["release_year"] <= end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(
        genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios


# Recommendation Page
def run_other_script():
    json_file = open('fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

# Load weights and them to model
    model.load_weights('fer.h5')

    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture(0)  # 0 indicates the default camera on your computer

    start_time = time.time()
#while (time.time() - start_time) < 10:

    while (time.time() - start_time) <= 50:
        ret, img = cap.read()
    
        if not ret:
            break

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels =img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            predictions = model.predict(img_pixels)
        #"x = image.img_to_array(img)" to "x = img_to_array(img)"
            max_index = int(np.argmax(predictions))

            emotions = [ 'neutral','happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
            predicted_emotion = emotions[max_index]

            cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            resized_img = cv2.resize(img, (1000, 700))
            cv2.imshow('Facial Emotion Recognition', resized_img)
            if(time.time() - start_time)>20:
                print("your emotion",predicted_emotion)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return predicted_emotion
def run_detection():
    predicted_emotion = run_other_script()
    st.write(f'Your mood is {predicted_emotion}')
    return predicted_emotion
global vd
def rec_page():
    
    st.header("RECOMMENDATION ENGINE")
    st.markdown("[Click here](https://www.example.com/) to run another script.")
    
    
     #  def call():
     #      return vd
    #u=call()   
    def fun():
      #  print("jkk",vd)
        #if vd=='neutral':
         #       vd=random.choice(genre)
        with st.container():
            col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
        with col3:
        #    for u in genre_names:
         #       if u==vd:
          #          break
            #if vd=='neutral':
             #   vd=random.choice(genre)
            st.markdown("***Choose your genre:***")
            genre = st.radio(
                "",
                genre_names, index=genre_names.index(vd))
        with col1:
            st.markdown("***Choose features to customize:***")
            start_year, end_year = st.slider(
                'Select the year range',
                1990, 2019, (2015, 2019)
            )
            acousticness = st.slider(
                'Acousticness',
                0.0, 1.0, 0.5)
            danceability = st.slider(
                'Danceability',
                0.0, 1.0, 0.5)
            energy = st.slider(
                'Energy',
                0.0, 1.0, 0.5)
            instrumentalness = st.slider(
                'Instrumentalness',
                0.0, 1.0, 0.0)
            valence = st.slider(
                'Valence',
                0.0, 1.0, 0.45)
            tempo = st.slider(
                'Tempo',
                0.0, 244.0, 118.0)
            tracks_per_page = 12
            test_feat = [acousticness, danceability,
                         energy, instrumentalness, valence, tempo]
            uris, audios = n_neighbors_uri_audio(
                genre, start_year, end_year, test_feat)
            tracks = []
            for uri in uris:
                track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(
                    uri)
                tracks.append(track)
        if 'previous_inputs' not in st.session_state:
            st.session_state['previous_inputs'] = [
                genre, start_year, end_year] + test_feat
        current_inputs = [genre, start_year, end_year] + test_feat

        if current_inputs != st.session_state['previous_inputs']:
            if 'start_track_i' in st.session_state:
                st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

        if 'start_track_i' not in st.session_state:
            st.session_state['start_track_i'] = 0

        with st.container():
            col1, col2, col3 = st.columns([2, 1, 2])
        if st.button("Recommend More Songs"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']
            : st.session_state['start_track_i'] + tracks_per_page]
        current_audios = audios[st.session_state['start_track_i']
            : st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                if i % 2 == 0:
                    with col1:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See more details"):
                            df = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                            fig = px.line_polar(
                                df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)

                else:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See more details"):
                            df = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                            fig = px.line_polar(
                                df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)

        else:
            st.write("No songs left to recommend")

        st.code("Algorithms that I have used in this filtering are the k-nearest neighbours and Random Forest")       



        st.subheader("Graph Representing Audio Features Importance")
        random_forest_audio_importance = Image.open('images/random_forest_audio_importance_feature.jpg')
        st.image(random_forest_audio_importance, caption ="random_forest_audio_feature_importance", width = 900)
  #  if st.button("Click for next"):
  #      fun()
    if st.button('Run Mood Detection'):
      vd=run_detection()  
      if vd=='neutral':
       vd=random.choice(genre_names)      
      fun()
# Home Page

def home_page():
    st.subheader('About Me')
    
    
    col1, col2 = st.columns(2)

    with col1:
        st.write(
        'Hi Microsoft, and this is my WebApp for Microsoft engage 2022 Program. I am in 2nd Year and pursuing BTech IT from Akhilesh Das Gupta Institute of Technology and Management, New Delhi India.')
        st.write('  Knowledgeable in the Web Application, services and product management. Motivated to gain more industrial experience with a growth oriented and technically advanced organizations.')
        st.write("Check out my [Github Repository](https://github.com/uditkatyal/songfitt_)")
        

    #with col2:
     #   image = Image.open(
      #  'images/img1.jpg')
       # st.image(image, caption='Udit Katyal', width=300)


# Conclusion Page

def conclusions_page():

    st.header('Conclusion ')
    st.subheader("Model Performance Summary")
    st.success("Accuracy Test Results")
    algo_accuracy = Image.open(
        'images/algos_accuracy.png')
    st.image(algo_accuracy, width=400)
    st.write('Using a dataset of 228, 000 Spotify Tracks, I was able to predict popularity(greater than 57 popularity) using audio-based metrics such as key, mode, and danceability without external metrics such as artist name, genre, and release date. The Random Forest Classifier was the best performing algorithm with 92.0 % accuracy and 86.4 % AUC. The Decision Tree Classifier was the second best performing algorithm with 87.5 % accuracy and 85.8 % AUC.')

    st.write('Moving forward, I will use a larger Spotify database by using the Spotify API to collect my own data, and explore different algorithms to predict popularity score rather than doing binary classification.')

    algo_auc = Image.open(
        'images/models_auc_area_under_curve.png')
    st.image(algo_auc, width=400)
    st.write("The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classess. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.")
     
    st.subheader("Music Trend Analysis")
    '''
    - Dataset is imbalanced with more Europian countries.
    - Few songs have managed to make in 96% of Top Charts of all countries.
    - Even though few artists have many occurances in the Top charts, they don't have any song in Top 10 tracks occurances.
    - Average song duration preferred by most is around 3:00 minutes to 3:20 minutes
    - People in Asian countries prefer longer song duration. Europian countries mostly listen to songs close to or less than 3 minutes.
    - Asian countries prefer less of explicit songs, only 18%, compared to world average of 35%. "Global Top 50 Chart" has 23 explicit songs.
    '''
    



st.session_state.app_mode = 'recommend'

def main():
    
    spr_sidebar()
    st.header("eSangeet (eS)")
    st.markdown(
        '**eSangeet** is a online Robust Music Recommendation Engine where in you can finds the best songs that suits your taste.  ') 
    st.markdown('Along with the rapid expansion of digital music formats, managing and searching for songs has become signiﬁcant. The purpose of this project is to build a recommendation system to allow users to discover music based on their listening preferences. Therefore in this model I focused on the public opinion to discover and recommend music.')    
    
    if st.session_state.app_mode == 'dataset':
        dataset_page()
    
    if st.session_state.app_mode == 'algo':
        algo_page()

    if st.session_state.app_mode == 'recommend':
        rec_page()

    if st.session_state.app_mode == 'report':
        report_page()

    if st.session_state.app_mode == 'conclusions':
        conclusions_page()

    if st.session_state.app_mode == 'home':
        home_page() 
    spr_footer()

# Run main()
if __name__ == '__main__':
    main()


