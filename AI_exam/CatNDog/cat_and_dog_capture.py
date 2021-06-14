import time

import cv2

# 객체생성 및 크기 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

flag = True
while flag:
    # 카메라로부터 화면 캡쳐
    ret, frame = capture.read() # frame이 이미지
    cv2.imshow("VideoFrame", frame) # 이미지 출력
    time.sleep(1) # 0.5초동안 중지
    print('capture') # 콘솔 창에도 출력
    key = cv2.waitKey(33) # 매개변수로 넘긴 시간안에 키 입력시 소스의 다음줄로 이동
    # esc 입력 시 while문 종료
    if key == 27:
        flag = False

capture.release() # VideoCapture 객체 종료
cv2.destroyAllWindow() # 창 닫기