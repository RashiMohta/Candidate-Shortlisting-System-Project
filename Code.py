# USAGE

# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4

# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

 

from scipy.spatial import distance as dist

from videoRotation import Rotation

from imutils.video import FileVideoStream

from imutils.video import VideoStream

from imutils import face_utils

import numpy as np

#import argparse

import imutils

import time

import dlib

import cv2

import csv

import videoRotation

from datetime import datetime

import face_recognition

import os

import subprocess as sp

import sys

import speech_recognition as sr

from os import path

from pydub import AudioSegment

from langdetect import detect

from moviepy.config import get_setting

from moviepy.tools import subprocess_call

 

 

directory=os.fsencode(r"C:\Users\aakash1.mal\Documents\Python_Recording_sessions\ML_Project\New_Videos")

dir=r"C:\Users\aakash1.mal\Documents\Python_Recording_sessions\ML_Project\New_Videos"

for file in os.listdir(directory):

    file=file.decode('utf-8')

    newFileName=Rotation(dir+"\\"+file)

    if newFileName=='empty':

        newFileName1=(dir+"\\"+file)

        videoRotated = "Video is not Rotated"

        cap = cv2.VideoCapture(newFileName1)

    else:

        cap = cv2.VideoCapture(newFileName)

        videoRotated = "Video is Rotated"

    #cap = cv2.VideoCapture(newFileName)

 

    def eye_aspect_ratio(eye):

           # compute the euclidean distances between the two sets of

           # vertical eye landmarks (x, y)-coordinates

           A = dist.euclidean(eye[1], eye[5])

           B = dist.euclidean(eye[2], eye[4])

   

           # compute the euclidean distance between the horizontal

           # eye landmark (x, y)-coordinates

           C = dist.euclidean(eye[0], eye[3])

   

           # compute the eye aspect ratio

           ear = (A + B) / (2.0 * C)

   

           # return the eye aspect ratio

           return ear

   

    

    

    #inputVideofile="C:/Users/aakash1.mal/Documents/Python_Recording_sessions/ML_Project/successful/candidate3.mp4"

    shapepredPath="C:/Users/aakash1.mal/Documents/Python_Recording_sessions/Machine_Learning/blink-detection/shape_predictor_68_face_landmarks.dat"

   

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

   

    # define two constants, one for the eye aspect ratio to indicate

    # blink and then a second constant for the number of consecutive

   # frames the eye must be below the threshold

    EYE_AR_THRESH = 0.31999

    EYE_AR_CONSEC_FRAMES = 7

   

    # initialize the frame counters and the total number of blinks

    COUNTER = 0

    TOTAL = 0

   

    # initialize dlib's face detector (HOG-based) and then create

    # the facial landmark predictor

    print("[INFO] loading facial landmark predictor...")

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(shapepredPath)

   

    # grab the indexes of the facial landmarks for the left and

    # right eye, respectively

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

   

    # start the video stream thread

    print("[INFO] starting video stream thread...")

 

   

    fileStream = True

    #vs = VideoStream(src=0).start()

    #vs = VideoStream(usePiCamera=True).start()

    #fileStream = False

    time.sleep(1.0)

    checkFrame=True

    # loop over frames from the video stream

    while True:

           # if this is a file video stream, then we need to check if

           # there any more frames left in the buffer to process

           #if fileStream and not vs.more():

            #      break

   

           # grab the frame from the threaded video file stream, resize

           # it, and convert it to grayscale

           # channels)

           #frame = vs.read()     

           now = datetime.now()

        #currentDate = str(now.month) + "_" + str(now.day) + "_" + str(now.year) + "_" + str(now.time)

           currentDate = datetime.now().strftime("%Y_%m_%d-%H_%M_%S") 

    

           ret, frame = cap.read()

           if ret is False:

            break

           else:

               if checkFrame==True:

                        videoImage=frame

                        cv2.imwrite(r"C:\Users\aakash1.mal\Documents\Python_Recording_sessions\Output_File\Img_" + currentDate + ".jpg",videoImage)

                        checkFrame=False

                        image_to_be_matched = face_recognition.load_image_file('C:/Users/aakash1.mal/Documents/Python_Recording_sessions/ML_Project/successful/Pic_2.png')

                        image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]

                        current_image = face_recognition.load_image_file(r"C:\Users\aakash1.mal\Documents\Python_Recording_sessions\Output_File\Img_"+ currentDate + ".jpg")

                        current_image_encoded = face_recognition.face_encodings(current_image)[0]

                        result = face_recognition.compare_faces([image_to_be_matched_encoded],current_image_encoded,tolerance=0.6)

                        if result[0] == True:

                           print("Matched" )

                           faceMatched="Matched"

                        else:

                           print("Not Matched" )

                           faceMatched="Not Matched"

        #if frame

           frame = imutils.resize(frame, width=450)   

           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   

           # detect faces in the grayscale frame

   

           faces = faceCascade.detectMultiScale(gray,

        #scaleFactor=1.3,

        #minNeighbors=3

        #minSize=(30, 30)

        )

           #print(faces.shape)

         

           

           

    

           rects = detector(gray, 0)

   

           # loop over the face detections

           for rect in rects:

                  # determine the facial landmarks for the face region, then

                  # convert the facial landmark (x, y)-coordinates to a NumPy

                  # array

                  shape = predictor(gray, rect)

                  shape = face_utils.shape_to_np(shape)

   

                  # extract the left and right eye coordinates, then use the

                  # coordinates to compute the eye aspect ratio for both eyes

                  leftEye = shape[lStart:lEnd]

                  rightEye = shape[rStart:rEnd]

                  leftEAR = eye_aspect_ratio(leftEye)

                  rightEAR = eye_aspect_ratio(rightEye)

   

                  # average the eye aspect ratio together for both eyes

                  ear = (leftEAR + rightEAR) / 2.0

   

                  # compute the convex hull for the left and right eye, then

                  # visualize each of the eyes

                  leftEyeHull = cv2.convexHull(leftEye)

                  rightEyeHull = cv2.convexHull(rightEye)

                  cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)

                  cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

   

                  # check to see if the eye aspect ratio is below the blink

                  # threshold, and if so, increment the blink frame counter

                  if ear < EYE_AR_THRESH:

                         COUNTER += 1

   

                  # otherwise, the eye aspect ratio is not below the blink

                  # threshold

                  else:

                         # if the eyes were closed for a sufficient number of

                         # then increment the total number of blinks

                         if COUNTER >= EYE_AR_CONSEC_FRAMES:

                               TOTAL_Blinks += 1

    

                         # reset the eye frame counter

                         COUNTER = 0

   

                  # draw the total number of blinks on the frame along with

                  # the computed eye aspect ratio for the frame

                  cv2.putText(frame, "Blinks: {}".format(TOTAL_Blinks), (10, 30),

                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                  cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),

                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

           # show the frame

           cv2.imshow("Frame", frame)

           key = cv2.waitKey(2) & 0xFF

       # if the `q` key was pressed, break from the loop

           if key == ord("q"):

              break

    print("Total Blinks are :" + str(TOTAL_Blinks))

    print("Number of faces detected: " + str(faces.shape[0]))

 

    def ffmpeg_extract_audio(inputfile,output,bitrate=3000,fps=44100):

        """ extract the sound from a video file and save it in ``output`` """

        cmd = [get_setting("FFMPEG_BINARY"), "-y", "-i", inputfile, "-ab", "%dk"%bitrate,"-ar", "%d"%fps, output]

        subprocess_call(cmd)

 

    ffmpeg_extract_audio(newFileName1,r"C:\Users\aakash1.mal\Documents\Python_Recording_sessions\Output_File\Audio_Output\Video_" + currentDate + ".wav")

    AUDIO_FILE = r"C:\Users\aakash1.mal\Documents\Python_Recording_sessions\Output_File\Audio_Output\Video_" + currentDate +".wav"

                               

    r = sr.Recognizer()

    with sr.AudioFile(AUDIO_FILE) as source:

            audio = r.record(source)

            text=r.recognize_google(audio)

            print("Done Processing:" +str(text))

            print(detect(text))

 

    def csv_Writer():

        with open('innovators.csv', 'w', newline='') as file:

             writer = csv.writer(file)

             writer.writerow(["SrNo", "VideoRotated", "TOTAL_Blinks","MultiFaceDetected","TextExtracted",])

             writer.writerow([counter, VideoRotated,TOTAL_Blinks,No_Of_Faces,text])

  

 

 

# do a bit of cleanup

cap.release()

cv2.destroyAllWindows()


 

Video Rotation:-

## RUNS VIDEO FINE AND Rotates Frame of video and then saves it in solution folder :

import numpy as np

import cv2

import imutils

import os

from datetime import datetime

import os

import face_recognition

 

def detectFace(image):   

    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,

    #minNeighbors=3

    #minSize=(30, 30)

    )

    No_Of_Faces = "Found {0} Faces.".format(len(faces))

    print(No_Of_Faces)

    return format(len(faces));

       

def Rotation(inputVideofile):

    cap = cv2.VideoCapture(inputVideofile)#C:/Users/aakash1.mal/Documents/Python_Recording_sessions/ML_Project/\New_Videos/Video_3.mp4')

    FinalAngle=0

    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret==True:   

            detected=detectFace(frame)

            

            if (int(detected)==0):

                for angle in np.arange(90,360,90):           

                    rot = imutils.rotate_bound(frame, angle)

                    cv2.imshow("Angle", rot)

                    if(int(detectFace(rot))):

                        FinalAngle=angle

                        break

            else:

                FinalAngle=0

                file_output='empty'

                break

 

            if FinalAngle!=0:

                print("Rotation success at angle "+ str(FinalAngle))

                break           

        if ret==False:

            break   

        

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    if FinalAngle!=0:

   

        #Convert whole video to FinalAngle and use it further for Eye-blinking process

        frame_width = int(cap.get(3))

        frame_height = int(cap.get(4))

        fw = int(cap.get(3))

        fh = int(cap.get(4))

        #print("w,h",fw,fh)

        #changed width to height and height to width for saving

        now = datetime.now()

        #currentDate = str(now.month) + "_" + str(now.day) + "_" + str(now.year) + "_" + str(now.time)

        currentDate = datetime.now().strftime("%Y_%m_%d-%H_%M_%S") 

        file_output = os.path.join(r"C:\Users\aakash1.mal\Documents\Python_Recording_sessions\Output_File\candidate_" + currentDate + ".mp4")

        out = cv2.VideoWriter(file_output ,cv2.VideoWriter_fourcc(*'FMP4'), 20.0, (frame_height,frame_width))

       

        while(True):

            ret, frame = cap.read()

            if ret == True:

                rot = imutils.rotate_bound(frame, FinalAngle)

                out.write(rot)

               

            else:

                break

       

        

        out.release()

        cap.release()

        cv2.destroyAllWindows()

    return file_output

 
