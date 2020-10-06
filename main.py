# -*- coding: utf-8 -*-
"""
@author: Dhruv Tongia
"""

# Face Emotion Recognition
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2

# the seven most common emotions of face
num_classes = 7
emotion_labels = ["Angry ", "Disgust ", "Fear ", "Happy ", "Sad ", "Surprise", "Neutral "]
emotion_labels=np.array(emotion_labels)
print(emotion_labels)



#input size of image to the model is(48,48,1)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.

def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
    
        # resizing the image according to the input requirement of the model
        
        img=cv2.resize(roi_gray,(48,48))
        img=img_to_array(img)
        img=img/255.0
        prediction = model.predict(img.reshape(1,48,48,1))
        #label of the predicted emotion
        label=emotion_labels[np.argmax(prediction)]
        #cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 255, 0), 2) # We draw a rectangle around the detected object.
        cv2.putText(frame,label , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2, cv2.LINE_AA) # We put the label of the class right above the rectangle.
        
        
    return frame # We return the image with the detector rectangles.


model=load_model('Fer2013_cnn.h5')

video_capture = cv2.VideoCapture(0) # We turn the webcam on.

while True: # We repeat infinitely (until break):
    _, frame = video_capture.read() # We get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    canvas = detect(gray, frame) # We get the output of our detect function.
    cv2.imshow('Video', canvas) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.