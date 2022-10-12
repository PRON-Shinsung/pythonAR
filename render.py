import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def create_image(h, w, d):
    image = np.zeros((h, w,  d), np.uint8)
    color = tuple(reversed((0,0,0)))
    image[:] = color
    return image

def create_image_multiple(h, w, d, hcout, wcount):
    image = np.zeros((h*hcout, w*wcount,  d), np.uint8)
    color = tuple(reversed((0,0,0)))
    image[:] = color
    return image


def showMultiImage(dst, src, h, w, d, col, row):
    if d == 3:
        dst[(col*h):(col*h)+h, (row*w):(row*w)+w] = src[0:h, 0:w]

    elif d == 1:
        dst[(col*h):(col*h)+h, (row*w):(row*w)+w, 0] = src[0:h, 0:w]
        dst[(col*h):(col*h)+h, (row*w):(row*w)+w, 1] = src[0:h, 0:w]
        dst[(col*h):(col*h)+h, (row*w):(row*w)+w, 2] = src[0:h, 0:w]


while True:
    ret, frame = cap.read()

    frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    height, width, depth = frame.shape[0], frame.shape[1], frame.shape[2]

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_3chan = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, gray = cv2.threshold(gray, 190, 255, cv2.THRESH_TOZERO)
    threshold_3chan = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.edgePreservingFilter(gray, flags=1, sigma_s=45, sigma_r=0.2)

    filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    _, cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.contourArea(cnts[0])

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for i in cnts:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            size = len(screenCnt)
            break
        else:
            size = 0

    if (size > 0):
        cv2.line(frame, tuple(screenCnt[0][0]), tuple(screenCnt[size-1][0]), (255, 0, 0), 3)
        for j in range(size-1):
            color = list(np.random.random(size=3), 255)
            cv2.line(frame, tuple(screenCnt[j, 0]), tuple(screenCnt[j-1, 0]), color, 3)


    dstimage = create_image_multiple(height, width, depth, 2, 2)

    showMultiImage(dstimage, frame, height, width, depth, 0, 0)
    showMultiImage(dstimage, gray_3chan, height, width, depth, 0, 1)
    showMultiImage(dstimage, threshold_3chan, height, width, depth, 1, 0)
    showMultiImage(dstimage, filtered, height, width, depth, 1, 1)

    cv2.imshow('frame', dstimage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
