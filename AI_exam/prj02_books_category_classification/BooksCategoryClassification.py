"""
books category classification GUI
tensorflow version: 2.3.0
"""
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import *
from tensorflow.keras.models import load_model

# ui 파일 로드
form_window = uic.loadUiType("./mainwindow.ui")[0]

# QWidget과 ui 파일을 상속받은 클래스
class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()  # 부모의 생성자 실행
        self.path = None
        self.setupUi(self)
        self.model = load_model("./models/books_classification_0.9008.h5")
        # 클릭 시그널이 발생하면 predict_image 함수 실행
        self.reset_1.clicked.connect(self.clearText)
        # self.reset_2.clicked.connect(self.clearText)

    def clearText(self):
        self.InputText.clear()


# 현재 py 파일의 절대경로를 인수로 QApplication 객체 생성
app = QApplication(sys.argv)

# ui 파일을 상속받은 QWidget 클래스를 띄움
mainWindow = Exam()
mainWindow.show()

# AQpplication 객체에서 이벤트 루프를 실행하고, 윈도우 x 버튼 클릭 시 종료함
sys.exit(app.exec_())
