"""
이미지 파일을 바이너리로 train, test 나눠 저장
"""
import glob

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

img_dir = '../datasets/horse-or-human/'  # 데이터 폴더 경로
categories = ['horses', 'humans']  # 이미지 카테고리

# 이미지 크기 설정
image_w = 64
image_h = 64

# image, label 저장할 변수 초기화
X = []
Y = []
files = None

# 카테고리별 반복(2회)
for idx, category in enumerate(categories):
    # 경로 내 해당 카테고리로 시작하는 폴더 내 파일 경로 리스트
    files = glob.glob(img_dir + category + '/*.png')
    # 각 경로명마다 반복
    for i, f in enumerate(files):
        # 파일 관련 작업이므로 try-except 사용
        try:
            # 이미지를 열어 RGB로 바꾸고 resize
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize((image_w, image_h))  # 매개변수 튜플로
            # 이미지를 ndarray로 저장
            data = np.asarray(img)
            # 값 list에 저장
            X.append(data)
            Y.append(idx)
            # 300번마다 진행상황 출력
            if i % 300 == 0:
                print(f'{category}: {f}')

        # 오류 발생 시 출력
        except:
            print(f'{category} {i} 번째에서 에러')

# ndarray로 변환
X = np.array(X)
Y = np.array(Y)

# 스케일링
X = X / 255
print(X[0])
print(Y[:5])

# train, test 나눔
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1,
)

# 파일로 저장
xy = (X_train, X_test, Y_train, Y_test)
np.save('../datasets/horse-or-human/binary_image_data.npy', xy)
