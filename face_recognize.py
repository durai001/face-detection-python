import face_recognition
import cv2
import numpy as np
import os
video_capture = cv2.VideoCapture(0)
  
known_face_encodings = [ ]
known_face_names = []
# import faceDetection cascase classifier
face_cascade = cv2.CascadeClassifier('./faceDetection.txt') 
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while 1:
    ret,img = video_capture.read()
  
    if video_capture is None or not video_capture.isOpened():
        print("camera not found")
        break
    elif cv2.waitKey(1) & 0xFF == ord('d'):
         break    
    else:
        #copy image to new variable to save the detected image (*** note its not require)
        imgCopy=img.copy()
        name=""
        #to get bg to gray scal image and detected  faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    
        for (x,y,w,h) in faces: 
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
            roi_gray = gray[y:y+h, x:x+w] 
            roi_color = img[y:y+h, x:x+w] 
     
        cv2.imshow("Take picture",img)         
        k = cv2.waitKey(33)
        if(k==115):
            #get user name from terminal input 
            name = input("Enter user name: ")
            known_face_encodings.append(face_recognition.face_encodings(imgCopy)[0])
            known_face_names.append(name)
            cv2.imwrite(name+".jpg",imgCopy) 
            continue
        # press q to close webcams
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        continue  
    
    
while True:
    ret, frame = video_capture.read()
    if video_capture is None or not video_capture.isOpened():
        break
    else:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
    
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Person detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()