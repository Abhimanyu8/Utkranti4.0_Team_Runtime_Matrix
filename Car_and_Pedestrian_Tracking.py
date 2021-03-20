import cv2

#Our Image
img_file = 'car.jpg'
#video = cv2.VideoCapture('Tesla Autopilot Dashcam Compilation 2018 Version.mp4')
video = cv2.VideoCapture('Pedestrians Compilation_Trim.mp4')

#Our car classifier and pedestriann classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#Create Classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


#Run Forever until the car stops
while True:

    #Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        #Must Convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect Cars and Pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw rectangles around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #Draw rectangles around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)


    #Display the image
    cv2.imshow('Tesla Recreated: Car and Pedestrian Detector', frame)

    key = cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release the VideoCapture object
video.release()


'''
#Create opencv image
img = cv2.imread(img_file)

#Convert to grayscale
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create Classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect Cars
cars = car_tracker.detectMultiScale(black_n_white)

#Draw rectangles
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

#Display the image
cv2.imshow('Tesla Recreated Car Detector', img)

cv2.waitKey()
'''