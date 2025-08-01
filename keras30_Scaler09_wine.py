from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

import time

dataset = load_wine()
x = dataset.data
y = dataset.target

# print(x.shape, y.shape)
#(178,13), (178,)
print(np.unique(y, return_counts=True))
print(x)
print(y)

# (array([0, 1, 2]), array([59, 71, 48]))
# [[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]
#  [1.320e+01 1.780e+00 2.140e+00 ... 1.050e+00 3.400e+00 1.050e+03]
#  [1.316e+01 2.360e+00 2.670e+00 ... 1.030e+00 3.170e+00 1.185e+03]
#  ...
#  [1.327e+01 4.280e+00 2.260e+00 ... 5.900e-01 1.560e+00 8.350e+02]
#  [1.317e+01 2.590e+00 2.370e+00 ... 6.000e-01 1.620e+00 8.400e+02]
#  [1.413e+01 4.100e+00 2.740e+00 ... 6.100e-01 1.600e+00 5.600e+02]]
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

# exit()


y = pd.get_dummies(y).values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, stratify=y)


scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)
x_test = scaler.transform(x_test)
# stratify is a parameter used in functions like train_test_split 
# from sklearn to ensure that the split maintains the same class distribution as 
# the original dataset.

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))



# exit()
model = Sequential()
model.add(Dense(32, input_dim = 13, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation = 'softmax'))


model.compile(loss = 'categorical_crossentropy',
              optimizer= 'adam', metrics=['acc'])

es = EarlyStopping(
    monitor= 'val_loss',
    mode='min',
    patience= 30,
    restore_best_weights=True
)



hist = model.fit(x_train, y_train,
                 epochs = 300, batch_size=8,
                 validation_split = 0.2,
                 callbacks=[es], verbose = 1)



loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss)
print('acc:', acc)


y_pred_probs = model.predict(x_test)

# Convert softmax outputs to predicted class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# scaler = MinMaxScaler()
# loss: 0.0006814672378823161
# acc: 1.0


# scaler = StandardScaler()
# loss: 1.0391674041748047
# acc: 0.4166666567325592


# scaler = MaxAbsScaler()
# loss: 1.117112159729004
# acc: 0.3888888955116272


# scaler = RobustScaler()
# loss: 1.096883773803711
# acc: 0.3333333432674408