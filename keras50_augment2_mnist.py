from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

datagen = ImageDataGenerator(
    # rescale=1./255,           #정규화 0~255 scaling.
    horizontal_flip=True,   #수평반전  <- 데이터 증폭 또는 변환 /  좌우반전
    # vertical_flip=True,       #수직반전 <- 데이터 증폭 또는 변환 / 상하반전
    width_shift_range=0.1,    #수치를 10% 이동 하겠다. 평행이동 10%
    # height_shift_range=0.1, #수직이동 10%
    rotation_range=15,      #360도 중 n도를 돌린다
    # zoom_range=1.2,         #1.2배 확대
    # shear_range=0.7,        #좌표점 고정 후 일부를 한쪽으로 끌어당기는 범위(찌부 만들기)
    fill_mode='nearest'     
)
# print(x_train[0].shape)

augment_size = 10000        # 6만개 -> 10만개로 만들거야.

randidx = np.random.randint(x_train.shape[0], size=augment_size)  # x_train의 60000의 사이즈는 40000이다
          # np.random.randint(60000, 40000)
print(randidx)                              # [17528 27056 39564 ... 52637 32955 52229]
print(np.min(randidx), np.max(randidx))     # 1 59998

x_augmented = x_train[randidx].copy()       # 4만개의 데이터 copy, copy로 새로운 메모리 할당.
                                            # 서로 영향 x

y_augmented = y_train[randidx].copy()

print(x_augmented)
print(x_augmented.shape)                    # (10000, 28, 28)
print(y_augmented.shape)                    # (10000,)


# x_augmented = x_augmented.reshape(40000, 28, 28, 1)
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 1,)
print(x_augmented.shape)                    # (10000, 28, 28, 1)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)                    # (10000, 28, 28, 1)

print(x_train.shape)                        # (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)          # (60000, 28, 28, 1) (10000, 28, 28, 1)


x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))    # shape 때문에 2개 맞추는 거임
print(x_train.shape, y_train.shape)         # (70000, 28, 28, 1) (70000,)


print(np.max(x_train), np.min(x_train)) # 1.0  0.0
print(np.max(x_test), np.min(x_test))   # 1.0  0.0

path = './_save/keras36_cnn5/'
model = load_model(path + 'k36_0610_1757_0242-0.0596.hdf5')

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=50, verbose=1,
                   restore_best_weights=True,
                   )

model.fit(x_train, y_train, epochs=500, batch_size=512, # batch는 행이다!!!!!!!!
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es],
                 )


loss= model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])


y_pred = model.predict(x_test)

y_test = y_test.values
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:' , round(acc, 4))

y_pred = (y_pred > 0.5).astype(int)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

images = x_test.reshape(-1, 28, 28, 1)  # x_test가 (N, 64)인 경우만 OK

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i])
    plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# loss :  0.04990960657596588
# acc: 0.9846