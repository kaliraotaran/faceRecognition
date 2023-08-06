import face_recognition
import os, sys
import cv2
import numpy as np
import math


def face_confidence(face_distance, face_match_threshold = 0.6):
    range = (1.0-face_match_threshold)
    linear_value = (1.0 -face_distance)/(range *2.0)

    if face_distance >face_match_threshold:
        return str(round(linear_value *100, 2)) + '%'
    else:
        value = (linear_value + ((1.0-linear_value) *math.pow((linear_value-0.5) *2, 0.2))) *100 
        return str(round(value, 2)) + '%'
    

class facerecognition:
    face_location=[]
    face_encoding=[]
    face_name=[]
    know_face_encodings=[]
    know_face_name=[]
    process_current_frame = True

    def __init__(self):
        self.encode_faces()


    #encode the faces 
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.know_face_encodings.append(face_encoding)
            self.know_face_name.append(image)
        print(self.know_face_name)


    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found  . . . . . ')
        
        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame =  cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:,:,::-1]
            # find all the faces in the frame
                self.face_location= face_recognition.face_locations(rgb_small_frame)
                self.face_encoding = face_recognition.face_encodings(rgb_small_frame, self.face_location)

                self.face_name=[]
                for face_encoding in self.face_encoding:
                    matches = face_recognition.compare_faces(self.know_face_encodings, face_encoding)
                    name = 'unknown'
                    confidence = 'unknown'

                    face_distances =face_recognition.face_distance(self.know_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.know_face_name[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                    self.face_name.append(f'{name} ({confidence}) ')

            self.process_current_frame  = not self.process_current_frame

            # display the annotations
            for(top, right, bottom, left), name in zip(self.face_location, self.face_name):
                top*= 4
                right +=4
                left *=4
                bottom *=4

                cv2.rectangle(frame, (left, top), {right, bottom}, (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom -35 ), {right, bottom}, (0,0,255), -1)
                cv2.putText(frame, name, (left +6, bottom +6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

            cv2.imshow('Face recognition', frame)

            if cv2.waitKey(100) == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()



if __name__ =='__main__':
    fr=facerecognition()
    fr.run_recognition()

