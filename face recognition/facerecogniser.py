import cv2

def drawrectangle(img,classifier,scalefactor,minneighbours,color,text):
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features=classifier.detectMultiScale(grey_img,scalefactor,minneighbours)

    coords=[]
    
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
        id, conf = clf.predict(grey_img[y:y + h, x:x + w])
        if(id==1 and conf<40):
            cv2.putText(img, "Harshit", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif (id == 2 and conf<40):
            cv2.putText(img, "Mom", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif (id == 3 and conf<40):
            cv2.putText(img, "Sakshi", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Random", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        coords = [x, y, w, h]
    return coords

def recognise(img,clf,classifier):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords=drawrectangle(img,classifier,1.1,4,color['blue'],'face')
    return img

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

video_capture=cv2.VideoCapture(0);

if not video_capture.isOpened():
    video_capture.open()

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

while(True):
    _, img= video_capture.read()
    img=recognise(img,clf,faceCascade)
    cv2.imshow("face recognition",img)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
