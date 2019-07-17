import cv2
import numpy as np
import os
from PIL import Image

# PIL is the Python Imaging Library which provides the python interpreter with image editing capabilities

def train_classifier(dir):
    face=[]
    id=[]
    path=[os.path.join(dir,f)  for f in os.listdir(dir)]

    for image in path:

        #reason for below special condition-
        #os.listdir() will give you every single file in the directory, including hidden files like .DS_Store.
        # In macOS, .DS_Store is a hidden file (any file starting with a . is hidden from Finder). In any case,
        # you just need to not try and read that as an image file.

        if image == os.path.join(dir,'.DS_Store') :
            continue

        img=Image.open(image).convert('L')

        # L mode will convert image to black and white

        img=np.array(img,'uint8')  #converting to numpy array
        face.append(img)
        current_id=os.path.split(image)[1].split('.')[0]

        id.append(int(current_id))


    # converting to numpy array
    id=np.array(id)

    # Train and save classifier
    clf= cv2.face_LBPHFaceRecognizer.create() # Local binary patterns histograms (LBPH) Face Recognizer
    clf.train(face,id)

    #training our custom classifier on images in face where target variables are stored in id

    clf.write("classifier.xml")


train_classifier("data")