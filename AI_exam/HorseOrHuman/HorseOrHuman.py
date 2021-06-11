"""
Cat and Dog GUI
"""
import sys

from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from tensorflow.keras.models import load_model

# 코드 실행 시 기존 파일과 동일한 전처리 과정을 가지므로 import해서 재사용
from horse_or_human_predict import preprocess

# ui 파일 로드
form_window = uic.loadUiType('./mainWidget.ui')[0]


# QWidget과 ui 파일을 상속받은 클래스
class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()  # 부모의 생성자 실행
        self.path = None
        self.setupUi(self)
        self.model = load_model('../models/horse_or_human_binary_classification.h5')
        # 클릭 시그널이 발생하면 predict_image 함수 실행
        self.btn_select.clicked.connect(self.predict_image)

    def predict_image(self):
        # 윈도우 file chooser 사용해 이미지 파일 선택 / (경로, 선택 타입) 튜플 반환
        self.path = QFileDialog.getOpenFileName(
            self,
            "Open file", r'C:\Users\조석용\Documents\GitHub\study-dl\AI_exam\datasets\horse-or-human',
            # file chooser 경로 설정
            "Image Files(*.png);;AllFiles(*.*)"  # 확장자 선택 목록
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
                # horse_or_human_predict에서 사용했던 preprocess 함수 사용
                data = preprocess(self.path[0])
            except:
                print('error')

            # 모델 학습 및 결과 출력
            predict_value = self.model.predict(data)[0][0]
            print(predict_value)
            self.bar_human.setMaximum(100)
            self.bar_horse.setMaximum(100)
            self.bar_human.setValue((predict_value * 100).round())
            self.bar_horse.setValue(100-(predict_value * 100).round())
            # if predict_value >= 0.5:
            #     self.lbl_predict.setText(f'이 이미지는 {(predict_value * 100).round()}% 확률로 human입니다')
            # else:
            #     self.lbl_predict.setText(f'이 이미지는 {100 - (predict_value * 100).round()}% 확률로 horse입니다')


# 현재 py 파일의 절대경로를 인수로 QApplication 객체 생성
app = QApplication(sys.argv)

# ui 파일을 상속받은 QWidget 클래스를 띄움
mainWindow = Exam()
mainWindow.show()

# AQpplication 객체에서 이벤트 루프를 실행하고, 윈도우 x 버튼 클릭 시 종료함
sys.exit(app.exec_())
