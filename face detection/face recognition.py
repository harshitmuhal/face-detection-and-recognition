import cv2

def drawrectangle(img,classifier,scalefactor,minneighbours,color,text):
    grey_img=cv2.cvtColor(grey_img,cv2.COLOR_BGR2GRAY)
    features=classifier.detectMultiScale(img,scalefactor,minneighbours)

    coords=[]
    
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img,classifier,mouthCascade,noseCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords=drawrectangle(img,classifier,1.1,4,color['blue'],'face')
    if len(coords)==4:
        cropped_img=img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        coords = drawrectangle(cropped_img, noseCascade, 1.1, 4, color['green'], "Nose")
        coords = drawrectangle(cropped_img, mouthCascade, 1.1, 15, color['white'], "Mouth")
    return img

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')
noseCascade =cv2.CascadeClassifier('nose.xml')

video_capture=cv2.VideoCapture(0);

if not video_capture.isOpened():
    video_capture.open()
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)


while(True):
    _, img= video_capture.read()
    img=detect(img,faceCascade,mouthCascade,noseCascade)
    cv2.imshow("face recognition",img)
    if cv2.waitKey(1) == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
