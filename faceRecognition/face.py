import face_recognition
import cv2
import numpy as np

print("Enter 1 to run the webcam")
print("Enter 2 to run an mp4 file of your choice")
flag = int(input())
if flag == 1:
    cam = cv2.VideoCapture(0)
elif flag == 2:
    print("Enter the mp4 file name enclosed in singel/double quotes")
    s = input()
    cam = cv2.VideoCapture('Video/' + s + '.mp4')
###################################################### Loading the images and storing there encodings #######################################################

aryamanImage = face_recognition.load_image_file("images/aryaman.jpg")
aryamanFaceEncoding = face_recognition.face_encodings(aryamanImage)[0]
aryamnImage1 = face_recognition.load_image_file("images/aryaman1.jpg")
aryamanFaceEncoding1 = face_recognition.face_encodings(aryamnImage1)[0]



knownFaceEncodings = [aryamanFaceEncoding, aryamanFaceEncoding1]
knonwFaceNames = ["Aryaman", "Aryaman"]

#############################################################################################################################################################

#Usefull variables
faceLocation = []
faceEncodings = []
faceNames = []
processThisFrame = True

while True:
    #Capturing the frame
    ret, frame = cam.read()
    #Resizing the frame for faster computation
    smallFrame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
    
    #Converting the frame from BGR, which openCV uses to RGB, which is used by face_recognition
    rgbSmallFrame = smallFrame[:,:,::-1]

    if processThisFrame:
        #Capturing the faces in the video and storing their encoding
        faceLocation = face_recognition.face_locations(rgbSmallFrame)
        faceEncodings = face_recognition.face_encodings(rgbSmallFrame, faceLocation)

        faceNames = []
        
        ############################# Comparing with the known encodings ###################################
        
        for face_encoding in faceEncodings:
            matches = face_recognition.compare_faces(knownFaceEncodings, face_encoding)
            name = "Unknow"

            faceDistances = face_recognition.face_distance(knownFaceEncodings, face_encoding)
            bestMatchIndex = np.argmin(faceDistances)

            if matches[bestMatchIndex]:
                name = knonwFaceNames[bestMatchIndex]
            
            faceNames.append(name)
        #######################################################################################################
    processThisFrame = not processThisFrame


    for (top, right, bottom, left), name in zip(faceLocation, faceNames):
        #Resizing the frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        ############################## Forming the rectangel ##################################################
        
        cv2.rectangle(frame, (left, top), (right, bottom), (200, 0, 122), 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (200, 0, 122), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame, name, (left + 6, bottom -6), font, 1.0, (255, 255, 255), 1)
        
        ########################################################################################################
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()










