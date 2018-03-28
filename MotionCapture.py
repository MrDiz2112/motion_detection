import numpy as np
import cv2
from collections import deque
from ImageOperations import ImageOperations as IOps

FRAMES_COUNT = 5
BUFFER = 50


def DrawRectangle(cnt, img):
    x, y, w, h = cv2.boundingRect(cnt)

    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 3, (255, 255, 0), 3)


def SelectRegion(event, x, y, flags, param):
    global rect_pt1, rect_pt2, drawing_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_pt1 = (x, y)
        drawing_mode = True
    elif drawing_mode and event == cv2.EVENT_MOUSEMOVE:
        rect_pt2 = (x, y)
    elif drawing_mode and event == cv2.EVENT_LBUTTONUP:
        rect_pt2 = (x, y)
        drawing_mode = False


rect_pt1 = (0, 0)
rect_pt2 = (0, 0)

drawing_mode = False

if __name__ == "__main__":
    cam = cv2.VideoCapture('Samples/traffic3.mp4')
    width = int(cam.get(3))
    height = int(cam.get(4))

    winName = "SUPER MEGA MOTION DETECTION"
    cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

    refImg = None
    framesQueue = []

    trajectoriesDeque = deque(maxlen=BUFFER)

    for j in range(FRAMES_COUNT):
        framesQueue.append(cam.read()[1])

    while True:
        # Кадр для показа
        frame = framesQueue[FRAMES_COUNT // 2]

        # Первый и последний кадры
        f0 = IOps.ConvertToGray(framesQueue[0])
        f1 = IOps.ConvertToGray(frame)
        f2 = IOps.ConvertToGray(framesQueue[FRAMES_COUNT - 1])
        movObject = IOps.CreateMovingObject(f0, f1, f2)

        # Контуры
        im2, contours, hierarchy = cv2.findContours(np.copy(movObject),
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_big = []

        for cnt in contours:
            if (cv2.contourArea(cnt) > 70):
                contours_big.append(cnt)
        contours_complete = IOps.ConnectNearbyContours(contours_big, 70)

        for cnt in contours_complete:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            if (min(rect_pt1[0], rect_pt2[0]) < cx < max(rect_pt1[0],
                                                         rect_pt2[0]) and
                    min(rect_pt1[1], rect_pt2[1]) < cy < max(rect_pt1[1],
                                                             rect_pt2[1])):
                DrawRectangle(cnt, frame)
                trajectoriesDeque.appendleft((cx, cy))

        for i in range(0, len(trajectoriesDeque)):
            if trajectoriesDeque[i] is None:
                continue

            cv2.circle(frame, trajectoriesDeque[i], 3, (0, 0, 255), -1)

        cv2.rectangle(frame, rect_pt1, rect_pt2, (255, 0, 255), 2)
        cv2.setMouseCallback(winName, SelectRegion)
        cv2.imshow(winName + " bin", movObject)
        cv2.imshow(winName, frame)

        # If pressed ESC exit
        key = cv2.waitKey(1)
        if key & 0xFF == ord(' '):
            cv2.waitKey(0)

        if key & 0xFF == 27:
            break

        framesQueue.pop(0)
        endOfFrames, new_frame = cam.read()
        
        if not endOfFrames:
            break
        framesQueue.append(new_frame)

    cam.release()
    cv2.destroyAllWindows()
