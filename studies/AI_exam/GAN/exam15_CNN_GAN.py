import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = "./CNN_OUT_img/"
img_shape = (28, 28, 1)
epochs = 10000
batch_size = 128
noise = 100
sample_intarval = 100

(X_train, _), (_, _) = mnist.load_data()  # 모델이 이미지를 새로 생성하므로 검증용 데이터 x
print(X_train.shape)

X_train = X_train / 127.5 - 1  # -1 ~ 1 사이의 값을 가지도록 스케일링
X_train = np.expand_dims(X_train, axis=3)  # 차원 추가
print(X_train.shape)

# build generator
generator_model = Sequential()

# Dense layer
generator_model.add(Dense(7 * 7 * 256, input_dim=noise))
generator_model.add(BatchNormalization())  # 각 픽셀의 값을 더한 경우 128까지 커질 수 있으므로 정규화 -> 지역 저점 무시
generator_model.add(
    LeakyReLU(alpha=0.01)
)  # 값에 음수가 있기 때문에 relu 대신 사용(음수에도 약간의 기울기 가짐) / activation function(layer x)
generator_model.add(Reshape((7, 7, 256)))

# CNN layer
generator_model.add(
    Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")  # 가로세로 2배씩 뻥튀기 후 conv
)  # Conv2DTranspose는 upsamping
generator_model.add(BatchNormalization())  # 각 픽셀의 값을 더한 경우 128까지 커질 수 있으므로 정규화 -> 지역 저점 무시
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding="same"))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding="same"))
generator_model.add(Activation("tanh"))

generator_model.summary()

# build discriminator
discriminator_model = Sequential()

# CNN Layer
discriminator_model.add(Conv2D(32, kernel_size=5, strides=2, padding="same", input_shape=img_shape))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dropout(0.3))

discriminator_model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dropout(0.3))

discriminator_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dropout(0.3))

# Dense layer
discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation="sigmoid"))

discriminator_model.summary()

discriminator_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

# build GAN
gan_model = Sequential()
gan_model.add(generator_model)  # generator는 단독으로 학습 x
gan_model.add(discriminator_model)
discriminator_model.trainable = False  # 학습 x(backward x)
gan_model.compile(loss="binary_crossentropy", optimizer="adam")
print(gan_model.summary())

# real, fake label 생성
real = np.ones((batch_size, 1))
print(real.shape)  # 1 (128, 1)

fake = np.zeros((batch_size, 1))
print(fake.shape)  # 0 (128, 1)

for itr in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)  # 0 ~ 60000까지 128개
    real_imgs = X_train[idx]  # 랜덤한 128개 이미지

    z = np.random.normal(size=(batch_size, noise))  # 정규분포 노이즈 (128, 100)
    fake_imgs = generator_model.predict(z)  # generator가 noise로 생성한 이미지

    # batch size만큼 한번 학습
    # discriminator모델은 trainable = True일 때 compile 했으므로 학습 가능
    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)  # real과 fake의 평균 loss, accuracy

    # gan_model에서 discriminator_model.trainable = False 이므로 discriminator 부분은 학습 x
    z = np.random.normal(size=(batch_size, noise))  # 정규분포 노이즈 (128, 100)
    gan_hist = gan_model.train_on_batch(z, real)  # fake img에 label은 real로

    if itr % sample_intarval == 0:  # 100번마다 출력
        print(f"{itr} - D loss: {d_loss:0.6f}, accuracy: {d_acc*100:0.2f}, G loss: {gan_hist:0.6f}")

        row = col = 4
        z = np.random.normal(size=(row * col, noise))  # (16, 100)
        fake_imgs = generator_model.predict((z))
        fake_imgs = 0.5 * fake_imgs + 0.5  # 0 ~ 1 사이의 값을 가지도록 스케일링
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                cnt += 1
        path = os.path.join(OUT_DIR, f"img-{itr+1}")
        plt.savefig(path)
        plt.close()
