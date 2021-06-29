import dlib  # detect library
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # tenforflow 버전 1.x 사용

detector = dlib.get_frontal_face_detector()  # 얼굴 찾아줌
sp = dlib.shape_predictor("./models/shape_predictor_5_face_landmarks.dat")  # 얼굴에서 점 5개 찾아주는 모델

img = dlib.load_rgb_image("./imgs/02.jpg")
plt.figure(figsize=(16, 10))
plt.imshow(img)
plt.show()

# 얼굴 위치 표시
img_result = img.copy()
dets = detector(img)  # 얼굴 위치 반환(여러개일 수 있음)

if len(dets) == 0:
    print("cannot find faces!")
else:
    # 얼굴 좌표에 정사각형 그림
    fig, ax = plt.subplots(1, figsize=(16, 10))

    # 얼굴이 여러개일 경우 각각 사각형 표시
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()

# 눈, 코 5개의 위치에 표시
fig, ax = plt.subplots(1, figsize=(16, 10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)  # 위에서 불러온 모델로 5개의 위치 예측
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor="r", facecolor="r")
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()

# 추출한 얼굴 plot
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
fig, axes = plt.subplots(1, len(faces) + 1, figsize=(20, 16))
axes[0].imshow(img)
# 추출한 얼굴이 여러개일경우 전부 plot
for i, face in enumerate(faces):
    axes[i + 1].imshow(face)
plt.show()
