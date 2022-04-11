import numpy as np
import cv2
# import dlib
from torchvision import transforms

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from light_cnn import LightCNN_29Layers_v2

img_prefix = 'data/samples/'
res_prefix = 'data/res/'
model_prefix = 'models/'

names = {'wangzhi':'Wang Zhi'}
threshold= 0.5
font = cv2.FONT_HERSHEY_DUPLEX

haarcascade = model_prefix + 'face_detector/haarcascade_frontalface_alt2.xml'
# create an instance of the Face Detection Cascade Classifier
face_detector = cv2.CascadeClassifier(haarcascade)

predictor_path = model_prefix + 'facial_landmark_detector/shape_predictor_68_face_landmarks.dat.bz2'
# landmark_predictor = dlib.shape_predictor(predictor_path)

color_id_img = cv2.imread(img_prefix + 'wangzhi_id.jpg', cv2.COLOR_BGR2RGB)
id_img = cv2.cvtColor(color_id_img, cv2.COLOR_BGR2GRAY)
color_room_img = cv2.imread(img_prefix + 'example2.jpg', cv2.COLOR_BGR2RGB)
room_img = cv2.cvtColor(color_room_img, cv2.COLOR_BGR2GRAY)

# Detect faces using the haarcascade classifier on the "grayscale image"
room_faces = face_detector.detectMultiScale(room_img)
room_face_bbox = room_faces[2]
# shape = landmark_predictor(room_img, face_bbox)
# print(shape)
room_image_cropped = room_img[room_face_bbox[1]:(room_face_bbox[1]+room_face_bbox[3]),
                room_face_bbox[0]:(room_face_bbox[0]+room_face_bbox[2])]
cv2.imwrite(res_prefix+ 'room_image_cropped.png', room_image_cropped)

id_faces = face_detector.detectMultiScale(id_img)
id_face_bbox = id_faces[0]
# shape = landmark_predictor(room_img, face_bbox)
# print(shape)
id_image_cropped = id_img[id_face_bbox[1]:(id_face_bbox[1]+id_face_bbox[3]),
                id_face_bbox[0]:(id_face_bbox[0]+id_face_bbox[2])]
cv2.imwrite(res_prefix+ 'id_image_cropped.png', id_image_cropped)


#face recognition
transform = transforms.Compose([transforms.ToTensor()])

model = LightCNN_29Layers_v2(num_classes=80013)
model.eval()
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(model_prefix+'face_recognizer/LightCNN_29Layers_V2_checkpoint.pth.tar')['state_dict'])
# print('model loaded')

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

input = torch.zeros(1, 1, 128, 128)
first_img = cv2.resize(id_image_cropped, (128, 128))
first_img = np.reshape(first_img, (128, 128, 1))
first_img = transform(first_img)
input[0, :, :, :] = first_img
input = input.cuda()
_, first_face_encoding = model(input)

second_img = cv2.resize(room_image_cropped, (128, 128))
second_img = np.reshape(second_img, (128, 128, 1))
second_img = transform(second_img)
input[0, :, :, :] = second_img
input = input.cuda()
_, second_face_encoding = model(input)

face_similarity = cos(first_face_encoding, second_face_encoding)
print(face_similarity)
if face_similarity > threshold:

    (x,y,w,d) = room_face_bbox
    cv2.rectangle(color_room_img,(x,y),(x+w, y+d),(255, 255, 255), 2)
    cv2.putText(color_room_img, names['wangzhi'], (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(color_room_img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig.savefig(res_prefix+ 'res.png', bbox_inches='tight', pad_inches=0)

