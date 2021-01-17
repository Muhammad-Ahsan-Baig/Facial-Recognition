"""
Muhammad Ahsan Baig's Code
"""
# Import the modules
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
#C:/Users/Muhammad Ahsan Baig/Downloads/pics/
#E:/X/AI PROJECT UPLOADS/pics/
#E:/X/AI PROJECT UPLOADS/facial_recognition-main/

#os.listdir() will get you everything that's in a directory - files and directories. If you want just files, you could either filter this down using os.path:
#C:/Users/Muhammad Ahsan Baig/Downloads/pics/
# data_path = 'E:/X/AI PROJECT UPLOADS/pics/'
data_path = 'C:/Users/Muhammad Ahsan Baig/Downloads/pics/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]   # complete image location
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # reading image in grayscale format 
    Training_Data.append(np.asarray(images, dtype=np.uint8))   # convert image to numpy array and saving in list
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)    # convert labels list to numpy array 

model = cv2.face.LBPHFaceRecognizer_create()  # Local binary patterns histograms (LBPH) Face Recognizer.The idea with LBPH is not to look at the image as a whole, but instead, try to find its local structure by comparing each pixel to the neighboring pixels.LBP faces are not affected by changes in light conditions.In the end, you will have one histogram for each face in the training data set. That means that if there were 100 images in the training data set then LBPH will extract 100 histograms after training and store them for later recognition. Remember, the algorithm also keeps track of which histogram belongs to which person.

model.train(np.asarray(Training_Data), np.asarray(Labels)) # training the model

print("Dataset Model Training Complete!!!!!")
# Loading HAAR face classifier
face_classifier = cv2.CascadeClassifier('D:/University/BSE 7-B/Artificial Intelligence Lab/AI Project/haarcascade_frontalface_default.xml')
#face_classifier = cv2.CascadeClassifier('E:/X/AI PROJECT UPLOADS/facial_recognition-main/haarcascade_frontalface_default.xml')

def face_detector(img):
    # We use cvtColor, to convert image into grayscale format
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)  # image is taken in grayscale format because many image data manipulations are done in it

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2) # 2 POINT TO DRAW RECTANGLE-VERTICES,RGB VALUES,THICKNESS
        roi = img[y:y+h, x:x+w]  # CROPING IMAGE
        roi = cv2.resize(roi, (200,200)) # cv2.resize(image,(width,height))

    return img,roi
# Open Webcam
cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        result = model.predict(face)
        # Tell about the confidence of user.
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))


        # If confidence is greater than 80 then the face will be recognized.
        if confidence > 80:
            cv2.putText(image, "Muhammad Ahsan Baig", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) # writing the Name in the screen at x,ycoordinate,with hershey fonts-ints,fontscale,color-RGB(WHITE),thickness
            cv2.imshow('Face Cropper', image)
            print("Door Opened")
        # If confidence is less than 90 then the face will not be recognized.
        else:
            cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2) # RGB -BLUE
            cv2.imshow('Face Cropper', image)
            print("Alarm Ringing --- Intruder Infront of Door No 23 on 5th Floor .")

    # Raise exception in case, no image is found
    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2) # RGB -RED
        cv2.imshow('Face Cropper', image)
        print("Door Closed")
        pass
    # Breaks loop when enter is pressed
    if cv2.waitKey(1)==13:   #13 is the Enter Key
        break

# Release and destroyAllWindows
cap.release()
cv2.destroyAllWindows()
