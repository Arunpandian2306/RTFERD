import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
#import pyttsx3

# Load the pre-trained model
classifier = load_model(r'C:\Users\arun7\Downloads\Facial Emotion Recognition and Detection\pages\model1.h5')

# Define the labels for the emotions
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Define the face classifier
face_classifier = cv2.CascadeClassifier(r'C:\Users\arun7\Downloads\Facial Emotion Recognition and Detection\pages\haarcascade_frontalface_default.xml')

# Define the Streamlit app title
st.set_page_config(page_title='Real Time-Facial Emotion Recognition and Detection', page_icon=':smiley:')

# Add Streamlit authentication
# st.sidebar_page_config(page_title='Enter Username and Password')
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username == "RTFERD" and password == "arun23":
    # Create a streamlit window to display the video stream
    st.title('Real Time-Facial Emotion Recognition and Detection')
    stframe = st.empty()

    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Define a function to update the video stream
    def update():
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_classifier.detectMultiScale(gray)

        # Loop over the faces and make predictions
        for (x,y,w,h) in faces:
            # Extract the face ROI
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)

            # Preprocess the face ROI
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]

                # Draw a rectangle and label on the frame
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
                cv2.putText(frame, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                # Initialize the text-to-speech engine
                #engine = pyttsx3.init()

                # Set the voice rate
                #engine.setProperty('rate', 150)

                # Speak the detected emotion label
                #engine.say(label)
                #engine.runAndWait()
            else:
                cv2.putText(frame, 'No Faces', (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Display the frame in the streamlit window
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)

    # Start the update loop using Streamlit's "while" feature
    while stframe:
        update()

        # Exit the loop if the user clicks the "Stop" button
        if not stframe:
            break

    # Release the video capture
    cap.release()

else:
    st.warning("Enter username and password.")
