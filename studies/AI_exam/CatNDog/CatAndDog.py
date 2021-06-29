"""
Cat and Dog GUI
tensorflow version: 1.14.0
"""
import sys

import numpy as np
from PIL import Image
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from tensorflow.keras.models import load_model

# ui 파일 로드
form_window = uic.loadUiType('./mainWidget.ui')[0]


# QWidget과 ui 파일을 상속받은 클래스
class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()  # 부모의 생성자 실행
        self.path = None
        self.setupUi(self)
        self.model = load_model('../models/cat_and_dog_binary_classification_1_14_0.h5')
        # 클릭 시그널이 발생하면 predict_image 함수 실행
        self.btn_select.clicked.connect(self.predict_image)

    def predict_image(self):
        # 윈도우 file chooser 사용해 이미지 파일 선택 / (경로, 선택 타입) 튜플 반환
        self.path = QFileDialog.getOpenFileName(
            self,
            "Open file", r'C:\Users\조석용\Documents\GitHub\study-dl\AI_exam\datasets\cat_dog\train',  # file chooser 경로 설정
            "Image Files(*.jpg);;AllFiles(*.*)"  # 확장자 선택 목록
        )
        print(self.path)

        # file chooser를 강제로 닫을경우 ('', '') 리턴
        # file을 선택했을 경우에만 아래 실행
        if self.path[0]:
            # 파일을 열어 lbl_image label의 pixmap을 해당 파일로 설정
            with open(self.path[0], 'r') as f:
                pixmap = QPixmap(self.path[0])  # 파일을 QPixmap으로
                self.lbl_image.setPixmap(pixmap)  # 해당 QPixmap을 label의 Pixmap으로 설정

            # 데이터 전처리
            try:
                # 이미지를 열어 RGB로 바꾸고 resize
                img = Image.open(self.path[0])
                img = img.convert('RGB')
                img = img.resize((64, 64))  # 매개변수 튜플로
                # 이미지를 ndarray로 저장
                data = np.asarray(img)
                # 데이터 scaling
                data = data / 255
                # 모델에 적용할 수 있도록 reshape
                data = data.reshape(1, 64, 64, 3)
            except:
                print('error')

            # 모델 학습 및 결과 출력
            predict_value = self.model.predict(data)[0][0]
            print(predict_value)
            if predict_value >= 0.5:
                self.lbl_predict.setText(f'이 이미지는 {(predict_value * 100).round()}% 확률로 Dog입니다')
            else:
                self.lbl_predict.setText(f'이 이미지는 {100 - (predict_value * 100).round()}% 확률로 Cat입니다')


# 현재 py 파일의 절대경로를 인수로 QApplication 객체 생성
app = QApplication(sys.argv)

# ui 파일을 상속받은 QWidget 클래스를 띄움
mainWindow = Exam()
mainWindow.show()

# AQpplication 객체에서 이벤트 루프를 실행하고, 윈도우 x 버튼 클릭 시 종료함
sys.exit(app.exec_())
