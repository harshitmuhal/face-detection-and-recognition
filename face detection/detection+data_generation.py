import cv2

def generate_dataset(img,user_id,img_id):
    cv2.imwrite("data/"+str(user_id)+"."+str(img_id)+".jpg", img)

def drawrectangle(img,classifier,scalefactor,minneighbours,color,text):
    
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features=classifier.detectMultiScale(img,scalefactor,minneighbours)

    coords=[]
    
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img,classifier,img_id):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords=drawrectangle(img,classifier,1.1,4,color['blue'],'face')

    if len(coords)==4:
        user_id=1;
        roi_image=img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]

    return img

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

video_capture=cv2.VideoCapture(0);

if not video_capture.isOpened():
    video_capture.open()

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

img_id=0
while(True):
    _, img= video_capture.read()
    img=detect(img,faceCascade,img_id)
    img_id+=1
    cv2.imshow("face recognition",img)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
