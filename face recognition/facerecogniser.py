import cv2

def drawrectangle(img,classifier,scalefactor,minneighbours,color,text):
    # Converting image to gray-scale
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features=classifier.detectMultiScale(grey_img,scalefactor,minneighbours)

    coords=[]
    #we have to iterate because there might be multiple faces in the img
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
        id, _ = clf.predict(grey_img[y:y + h, x:x + w])
        if(id==1):
            cv2.putText(img, "Harshit", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif (id == 2):
            cv2.putText(img, "Mom", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif (id == 3):
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

# Loading custom classifier to recognize
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

video_capture=cv2.VideoCapture(0);

if not video_capture.isOpened():
    video_capture.open() #if video_capture couldn't initialise the capture we initialise it manually

# Set properties. Each set function returns True on success (i.e. correct resolution)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

while(True):
    _, img= video_capture.read()
    img=recognise(img,clf,faceCascade)
    cv2.imshow("face recognition",img)
    if cv2.waitKey(1) == ord('q'):
        break

# The cv::waitKey(n) function in OpenCV is used to introduce a delay of n milliseconds while rendering
# images to windows. When used as cv::waitKey(0) it returns the key pressed by the user on the active
# window. This is typically used for keyboard input from user in OpenCV programs.


#different output in waitKey(0) and waitKey(1)-

# cv2.waitKey(1),I get a continuous live video feed but with cv2.waitKey(0),I get still images

# waitKey(0) will pause your screen because it will wait infinitely for keyPress on your keyboard and
# will not refresh the frame(cap.read()) using your WebCam. waitKey(1) will wait for keyPress for just 1
# millisecond and it will continue to refresh and read frame from your webcam using cap.read().

video_capture.release()   #to realease the webcam because my work is done now
cv2.destroyAllWindows()
