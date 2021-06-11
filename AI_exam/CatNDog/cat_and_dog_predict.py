"""
Cat and Dog model load and predict
"""
import glob

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 기존 저장한 모델 로드
model = load_model('../models/cat_and_dog_binary_classification.h5')
print(model.summary())

# 이미지 경로와 크기
img_dir = '../datasets/cat_dog/train/'
image_w = 64
image_h = 64


# 해당 카테고리 이미지 중 하나의 경로를 랜덤하게 뽑아오는 함수
def get_rand_image(category):
    files = glob.glob(img_dir + f'{category}*.jpg')
    sample_index = np.random.randint((len(files)))
    sample_path = files[sample_index]
    return sample_path


# 각 카테고리 이미지 중 하나를 랜덤하게 뽑아옴
dog_sample_path = get_rand_image('dog')
cat_sample_path = get_rand_image('cat')

print(dog_sample_path)
print(cat_sample_path)


# 파일의 경로를 받아 해당 파일을 전처리해 ndarray로 반환하는 코드
def preprocess(path):
    # 이미지를 열어 RGB로 바꾸고 resize
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))  # 매개변수 튜플로
    # 이미지를 ndarray로 저장
    data = np.asarray(img)
    # 데이터 scaling
    data = data / 255
    # 모델에 적용할 수 있도록 reshape
    data_reshape = data.reshape(1, 64, 64, 3)
    return data_reshape


try:
    # 각 경로에서 해당 파일을 전처리한 데이터를 받아옴
    dog_data = preprocess(dog_sample_path)
    cat_data = preprocess(cat_sample_path)
except:
    print('error')

# 예측 및 결과 확인
# 예측값이 0과 1 사이 값으로 나오기 때문에 반올림
print(f'dog data: {model.predict(dog_data).round()}')
print(f'cat data: {model.predict(cat_data).round()}')
