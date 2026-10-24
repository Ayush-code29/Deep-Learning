import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import Perceptron
# A simple linear classifier (single-layer neural network). Good for binary classification tasks.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# accuracy_score: Measures how many predictions are correct.
# classification_report: Shows precision, recall, and F1-score for each class.
# confusion_matrix: Shows where the model is making correct vs wrong predictions.

# -------------------------------
# TensorFlow / Keras (Deep Learning utilities)
# -------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Sequential
# Lets you build a neural network layer-by-layer in a linear fashion.

from tensorflow.keras.layers import Dense
# Fully connected layer (every neuron connects to every neuron in the next layer).

from tensorflow.keras.layers import Dropout
# Randomly drops some neurons during training to prevent overfitting.

from tensorflow.keras.utils import to_categorical
# 👉 Converts class labels (like 0, 1, 2) into one-hot encoded format (e.g., [1,0,0], [0,1,0]).
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('iris.csv')
print(df.info())
X = df.drop(columns=['variety'])
y = df['variety']
encoder = LabelEncoder()
y_int = encoder.fit_transform(y)
# print(y_int)
X_train, X_test, y_train, y_test = train_test_split(
  X, y_int, test_size=0.20, random_state=42, stratify=y_int)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
per = Perceptron(max_iter=1000,random_state=42)
per.fit(X_train_scaled,y_train)
y_pred_percep = per.predict(X_test_scaled)
acc_percep = accuracy_score(y_pred_percep,y_test)
# print("Accuracy of model using perceptron is",acc_percep)
y_train_categorical = to_categorical(y_train, num_classes = 3)
y_test_categorical = to_categorical(y_test, num_classes = 3)
# print(y_train_categorical)
model = Sequential([
    Dense(16,input_dim = 4,activation='relu'),
    Dense(8,activation='relu'),
    Dense(3,activation='softmax')
])
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(X_train_scaled,y_train_categorical,epochs = 100,batch_size = 8,validation_split = 0.2, verbose = 0)
loss,acc = model.evaluate(X_test_scaled,y_test_categorical,verbose = 0)
print("Model using ANN has accuracy and loss ",acc,loss)