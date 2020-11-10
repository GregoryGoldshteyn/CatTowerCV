import numpy as np
import cv2

def findCircles(img):
    img = cv2.medianBlur(img,3)
    
    min_red = np.array([100,0,0])
    max_red = np.array([255, 245, 245])

    mask = cv2.inRange(img, min_red, max_red)

    img = cv2.bitwise_and(img, img, mask=mask)
    
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cimg = cimg[400:,400:800]

    # HoughCircles params:
    #   img: the image
    #   f: algorithm
    #   blurr: factor by which image is blurred before checking
    #   min_dist: minimum distance between circles
    #   param_1: first threshold to pass. Lower values mean more false positives
    #   param_2: second threshold
    #   min_radius: smallest circle to find
    #   max_radius: largest circle to find
    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,30,param1=40,param2=20,minRadius=0,maxRadius=30)

    print(circles)

    if not isinstance(circles, type(None)):
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(255,255,255),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(255,255,255),3)
    
    # perform a naive attempt to find the (x, y) coordinates of
    # the area of the image with the largest intensity value
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cimg)
    # cv2.circle(cimg, maxLoc, 5, (255, 0, 0), 2)

    return cimg

def runFindCirclesWebcam():
    # Get webcam feed
    cap = cv2.VideoCapture(0)

    # Set up capture device
    # cap.set(3,800) # Width in px
    # cap.set(4,600) # Height in px

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        cframe = findCircles(frame)

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', cframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

runFindCirclesWebcam()