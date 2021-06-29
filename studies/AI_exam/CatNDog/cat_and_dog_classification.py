"""
Cat and Dog classification
tensorflow version: 1.14.0
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = np.load(
    '../datasets/cat_dog/binary_image_data.npy',
    allow_pickle=True # 원래의 타입 그대로 읽어옴(안하면 문자열로 옴 ㅜㅜ)
)
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'Y_train shape: {Y_train.shape}')
print(f'Y_test shape: {Y_test.shape}')

# 모델 생성
model = Sequential()
"""CNN"""
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 input_shape=(64, 64, 3),
                 padding='same',
                 activation='relu',
                 )
          )
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 input_shape=(64, 64, 3),
                 padding='same',
                 activation='relu',
                 )
          )
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64,
                 kernel_size=(3, 3),
                 input_shape=(64, 64, 3),
                 padding='same',
                 activation='relu',
                 )
          )
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # DNN에 전달하기 위해 1차원으로 만듦
"""DNN"""
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )
model.summary()

# 7번 넘게 val_accuracy가 좋아지지 않을 경우 중단
early_stopping = EarlyStopping(monitor='val_acc', patience=7)

# 모델 학습
fit_hist = model.fit(X_train, Y_train,
                     batch_size=64,
                     epochs=100,
                     validation_split=0.15,
                     callbacks=[early_stopping],
                     )
# 모델 저장
model.save('../models/cat_and_dog_binary_classification_1_14_0.h5')

# 학습 결과 확인
score = model.evaluate(X_test, Y_test)
print(f'Evaluation loss: {score[0]}')
print(f'Evaluation accuracy: {score[1]}')

plt.subplot(2,1,1)
plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.subplot(2,1,2)
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_acc'], label='val_accuracy')
plt.legend()
plt.show()