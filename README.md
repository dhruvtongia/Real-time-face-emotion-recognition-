# Real-time-face-emotion-recognition-

## Description

This project presents the real time facial expression recognition of seven most basic human expressions: ANGER, DISGUST, FEAR, HAPPY, NEUTRAL SAD, SURPRISE.

This model can be used for prediction of expressions of both still images and real time video. However, in both the cases we have to provide image to the model. In case of real time video the image should be taken at any point in time and feed it to the model for prediction of expression. The system automatically detects face using HAAR cascade then its crops it and resize the image to a specific size and give it to the model for prediction. The model will generate seven probability values corresponding to seven expressions. The highest probability value to the corresponding expression will be the predicted expression for that image.

#### Technologies

1. Keras 
2. Python 3
3. OpenCV
4. Tensorflow as keras backend

---
## Model Architechture

The model was trained in jupyter notebook in google colab environment and is saved in the file  Fer2013_cnn.h5 .


---
## How To Use

* Download the <strong>Fer2013_cnn.h5</strong> file which contains the trained model along with its weight . 
* Run the <strong>main.py</strong> file to see the program working for real time predictions

#### Dataset

* Used the kaggle dataset from [Facial Expression Recognition Challenge](https://www.kaggle.com/deadskull7/fer2013)

* The data consists of 48x48 pixel grayscale images of faces but flattened.The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The final test set, consists of another 3,589 examples. 

---


## Author 

<strong>Dhruv Tongia</strong> 



