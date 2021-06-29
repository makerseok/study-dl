"""
Cat and Dog capture
tensorflow version: 1.14.0
"""
import sys
import time

import cv2
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
        # 객체생성 및 크기 설정
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        flag = True
        while flag:
            # 카메라로부터 화면 캡쳐
            ret, frame = capture.read()  # frame이 이미지
            cv2.imshow("VideoFrame", frame)  # 이미지 출력(VideoFrame은 윈도우 제목)
            time.sleep(0.5)  # 0.5초동안 중지(없을 시 가장 빠른 속도로)
            print('capture')  # 콘솔 창에도 출력
            cv2.imwrite('./imgs/capture.png', frame)

            key = cv2.waitKey(33)  # 매개변수로 넘긴 시간안에 키 입력시 소스의 다음줄로 이동
            # esc 입력 시 while문 종료
            if key == 27:
                flag = False

            pixmap = QPixmap('./imgs/capture.png')  # 파일을 QPixmap으로
            self.lbl_image.setPixmap(pixmap)  # 해당 QPixmap을 label의 Pixmap으로 설정

            # 데이터 전처리
            try:
                # 이미지를 열어 RGB로 바꾸고 resize
                img = Image.open('./imgs/capture.png')
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

        capture.release()  # VideoCapture 객체 종료
        cv2.destroyAllWindows()  # 창 닫기


# 현재 py 파일의 절대경로를 인수로 QApplication 객체 생성
app = QApplication(sys.argv)

# ui 파일을 상속받은 QWidget 클래스를 띄움
mainWindow = Exam()
mainWindow.show()

# AQpplication 객체에서 이벤트 루프를 실행하고, 윈도우 x 버튼 클릭 시 종료함
sys.exit(app.exec_())
