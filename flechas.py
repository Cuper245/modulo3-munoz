import time
import sys
import cv2
import numpy as np

def main():
    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)

    if cap.isOpened(): # try to get the first frame
        rval, frame = cap.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = cap.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")
    cap.release()


if __name__ == "__main__":
    main()
