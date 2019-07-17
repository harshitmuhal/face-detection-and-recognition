import cv2
import numpy as np
import os
from PIL import Image

def train_classifier(dir):
    face=[]
    id=[]
    path=[os.path.join(dir,f)  for f in os.listdir(dir)]

    for image in path:
        if image == os.path.join(dir,'.DS_Store') :
            continue

        img=Image.open(image).convert('L')
        img=np.array(img,'uint8')
        face.append(img)
        current_id=os.path.split(image)[1].split('.')[0]

        id.append(int(current_id))

    id=np.array(id)
    clf= cv2.face_LBPHFaceRecognizer.create()
    
    clf.write("classifier.xml")

train_classifier("data")
