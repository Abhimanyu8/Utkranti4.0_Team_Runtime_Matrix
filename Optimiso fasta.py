#-v "final obostha.mp4"
from threading import Thread
import sys
import cv2
from queue import Queue

class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        #used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        # initialize the queue used to store frames read from the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

#!!!!!!! M A I N - B E L O W !!!!!!!!!

from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

#Our car classifier and pedestriann classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#Create Classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# construct the argument parse and parse the arguments to add video of soice
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
args = vars(ap.parse_args())

# start the file video stream thread and allow the buffer to start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

# start the FPS timer (debug/ delete later)
fps = FPS().start()

# loop over frames from the video file stream
while fvs.more():
    #grab the frame from the threaded video file stream, resize
    #it, and convert it to grayscale (while still retaining 3 channels)
    frame = fvs.read()
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    #detect Cars and Pedestrians
    cars = car_tracker.detectMultiScale(frame)
    pedestrians = pedestrian_tracker.detectMultiScale(frame)

    #Draw rectangles around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #Draw rectangles around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)


    #Display the image
    #cv2.imshow('AutoDrop Car and Pedestrian Tracking module', frame)

    key = cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break


    
    # display the size of the queue on the frame
    cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
    # show the frame and update the FPS counter
    cv2.imshow("Detecting Passengers and Cars...", frame)
    cv2.waitKey(1)
    fps.update()





















    
