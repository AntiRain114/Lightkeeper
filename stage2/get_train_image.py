import win32api
import win32con
import cv2 as cv
import numpy as np
import copy
import math

# Morphological opening operation
def open_binary(binary, x, y):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # Get the structured element of the image
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)  # Opening operation
    return dst

# Morphological closing operation
def close_binary(binary, x, y):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # Get the structured element of the image
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)  # Closing operation
    return dst

# Morphological erosion operation
def erode_binary(binary, x, y):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # Get the structured element of the image
    dst = cv.erode(binary, kernel)  # Erosion
    return dst

# Morphological dilation operation
def dilate_binary(binary, x, y):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # Get the structured element of the image
    dst = cv.dilate(binary, kernel)  # Dilation
    return dst

def body_detetc(frame):
    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)  # Convert to YCrCb color space
    lower_ycrcb = np.array([0, 135, 85])
    upper_ycrcb = np.array([255, 180, 135])
    mask = cv.inRange(ycrcb, lowerb=lower_ycrcb, upperb=upper_ycrcb)  # YCrCb mask
    return mask

def get_roi(frame, x1, x2, y1, y2):
    dst = frame[y1:y2, x1:x2]
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    return dst

if __name__ == "__main__":
    capture = cv.VideoCapture(0)
    m_0 = 1500
    m_1 = 1500
    m_2 = 1500
    m_3 = 1500
    m_4 = 1500
    m_5 = 1500
    m_6 = 1500
    m_7 = 1500
    m_8 = 1500
    m_9 = 1500

    while True:
        ret, frame = capture.read()
        roi = get_roi(frame, 100, 250, 100, 250)
        k = cv.waitKey(50)
        if k == 27:  # Press ESC to exit
            break
        elif k == ord('a'):
            filename = r"E:\sourcedit\0\{}.jpg".format(m_0)
            cv.imwrite(filename, roi)
            m_0 += 1
            print('Saving 0-roi image, current image count:', m_0)
        elif k == ord('s'):
            cv.imwrite(r"E:\sourcedit\1\%s.jpg" % m_1, roi)
            m_1 += 1
            print('Saving 1-roi image, current image count:', m_1)
        elif k == ord('d'):
            cv.imwrite(r"E:\sourcedit\2\%s.jpg" % m_2, roi)
            m_2 += 1
            print('Saving 2-roi image, current image count:', m_2)
        elif k == ord('f'):
            cv.imwrite(r"E:\sourcedit\3\%s.jpg" % m_3, roi)
            m_3 += 1
            print('Saving 3-roi image, current image count:', m_3)
        elif k == ord('g'):
            cv.imwrite(r"E:\sourcedit\4\%s.jpg" % m_4, roi)
            m_4 += 1
            print('Saving 4-roi image, current image count:', m_4)
        elif k == ord('h'):
            cv.imwrite(r"E:\sourcedit\5\%s.jpg" % m_5, roi)
            m_5 += 1
            print('Saving 5-roi image, current image count:', m_5)
        elif k == ord('j'):
            cv.imwrite(r"E:\sourcedit\6\%s.jpg" % m_6, roi)
            m_6 += 1
            print('Saving 6-roi image, current image count:', m_6)
        elif k == ord('k'):
            cv.imwrite(r"E:\sourcedit\7\%s.jpg" % m_7, roi)
            m_7 += 1
            print('Saving 7-roi image, current image count:', m_7)
        elif k == ord('l'):
            cv.imwrite(r"E:\sourcedit\8\%s.jpg" % m_8, roi)
            m_8 += 1
            print('Saving 8-roi image, current image count:', m_8)
        elif k == ord('z'):
            cv.imwrite(r"E:\sourcedit\9\%s.jpg" % m_9, roi)
            m_9 += 1
            print('Saving 9-roi image, current image count:', m_9)
        cv.imshow("roi", roi)
        cv.imshow("frame", frame)
        c = cv.waitKey(50)
        if c == 27:
            break
    cv.waitKey(0)
    capture.release()
    cv.destroyAllWindows()
