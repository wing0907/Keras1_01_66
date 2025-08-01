#  train.csv 와 new_test.csv로 count 예측

import numpy as np
import pandas as pd
# import sklearn as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# print(sk.__version__)


# exit()
# 1. 데이터
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

print(train_csv)                # [10886 rows x 11 columns]
print(new_test_csv)                 # [6493 rows x 10 columns]
print(train_csv.shape)          # (10886, 11)
print(new_test_csv.shape)           # (6493, 10)
print(submission_csv.shape)     # (6493, 2)

x = train_csv.drop(['count'], axis=1)       # [10886 rows x 10 columns]
y = train_csv['count']                      # (10886,)
print(x)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=361,
    )


scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min',                    # 최대값 max, 알아서 찾아줘: auto
    patience=50,                     # 이만큼 참을 거다. local minimal, global minimal
    restore_best_weights=True,       #EarlyStopping 의 Default는 False. 최적의 weight 때 멈춘다. 최소지점 save.
)



hist = model.fit(x_train, y_train, epochs=2000, batch_size=32,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])
y_predict = model.predict(x_test)

r2 = r2_score(y_test, results)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, results)

y_submit = model.predict(new_test_csv)

print(submission_csv)
submission_csv['count'] = y_submit
print(submission_csv)

# submission_csv.to_csv(path + 'sampleSubmission_0526_1426.csv', index=False )

print('[loss]:', loss)
print('[rmse]:', rmse)
print('[r2]:', r2)

# scaler 안쓴거
# [loss]: 0.0008660773746669292
# [rmse]: 0.02942919034174128
# [r2]: 0.9999999739530253


# scaler = MinMaxScaler()
# [loss]: 0.014192224480211735
# [rmse]: 0.11913060534202265
# [r2]: 0.9999995731774484


# scaler = StandardScaler()
# [loss]: 0.019708842039108276
# [rmse]: 0.1403877974410236
# [r2]: 0.9999994072666254


# scaler = MaxAbsScaler()
# [loss]: 0.012188375927507877
# [rmse]: 0.11039963148606144
# [r2]: 0.9999996334477345


# scaler = RobustScaler()
# [loss]: 0.008056542836129665
# [rmse]: 0.08975851167781683
# [r2]: 0.9999997577008191