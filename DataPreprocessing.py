
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten

train_data=pd.read_csv('train.csv')
X_train_data=train_data.iloc[:,1:].values
Y_train_data=train_data.iloc[:,0].values
test_data=pd.read_csv('test.csv').values
X_train_data = X_train_data.reshape(X_train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
input_shape = (28,28,1)

from sklearn.preprocessing import OneHotEncoder
onehot=OneHotEncoder()
Y_train_data=onehot.fit_transform(Y_train_data.reshape(-1,1)).toarray()

plt.imshow(X_train_data[0].reshape(28,28),cmap='gist_gray').show()

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2,2)))
