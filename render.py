import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(img, label, (pt1[0], pt1[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

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


    ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cont in contours:
        approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
        vtc = len(approx)

        if vtc == 4:
            setLabel(frame, cont, 'Rec')

    dstimage = create_image_multiple(height, width, depth, 1, 2)

    showMultiImage(dstimage, frame, height, width, depth, 0, 0)
    #showMultiImage(dstimage, gray_3chan, height, width, depth, 0, 1)
    #showMultiImage(dstimage, threshold_3chan, height, width, depth, 1, 0)
    showMultiImage(dstimage, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), height, width, depth, 0, 1)

    cv2.imshow('frame', dstimage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
