import cv2
import sys
import numpy as np
import os
from keras.utils import np_utils
from sudoku_solver import solveSudoku
from copy import deepcopy
import time as t

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from keras.models import load_model

def find_rect_pts(pts):
    new_pts = pts
    epsilon = 0
    #while it is not approximating enough
    while len(new_pts) > 4:
        epsilon += 1 
        new_pts = cv2.approxPolyDP(pts, epsilon, True)
    return new_pts


def find_board(image):

    #find the largest contour(hopefully the outer rim of the puzzle)
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, 
                                             cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    #find the extreme 4 points of the contour
    board_pts = find_rect_pts(max_contour).reshape((4,2)).tolist()

    #sort in C formation
    board_pts   = sorted(board_pts, key=lambda x: x[1])
    board_pts   = sorted(board_pts[:2], key=lambda x: -x[0]),\
                  sorted(board_pts[2:], key=lambda x:  x[0])
    board_pts = [*board_pts[0], *board_pts[1]]
    
    return board_pts

def find_digit(img):
    
    def perc_filled(cnt):
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        cropped = img[y:y+h,x:x+w]
        perc_filled =  cv2.countNonZero(cropped) / float(w*h)
        if w < 15 and h < 15 or w > 40 and h > 40:
            return -1
        if perc_filled < 0.2:
            return -1
        if abs(w-h) / min(w,h) > 3.2:
            return -1
        return perc_filled

    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)

    digit = max(contours, key=perc_filled)
    if perc_filled(digit) != -1:
        return digit
    raise Exception()


def main(argv):
    #get file args
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(argv) == 2:
        file_name = argv[1]

        #read in the image
        print("Loading image: {}".format(file_name))
        orig_image = cv2.imread(file_name)
        orig_image = cv2.resize(orig_image, (500,500))
    else:
        cap = cv2.VideoCapture(0)
        while True:
            _, image = cap.read()
            cv2.putText(image, "Press any button to capture", (50,50),
                        font, 1, (0,0,255), 3, cv2.LINE_AA )
            cv2.imshow('Camera', image)
            key = cv2.waitKey(1)
            if key != -1:
                break
        _, orig_image = cap.read() 
        cv2.destroyAllWindows()


    #cvt to grayscale and blur
    thresh = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.GaussianBlur(thresh, (7,7), 0)
    
    #threshold the image adaptively with shadow correction and invert
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                                cv2.THRESH_BINARY_INV,11,2)

    thresh = cv2.erode(thresh, (3,3))

    #find the corners of the board
    board_pts = find_board(thresh)

    #transform the board points to the target points (orthophoto)
    orig_pts = np.array(board_pts, dtype=np.float32)
    ortho_pts = np.array([[500, 0],[0,0],[0,500],[500,500]], dtype=np.float32) 
    H = cv2.getPerspectiveTransform(orig_pts, ortho_pts)
    warp = cv2.warpPerspective(thresh, H, (500,500))
    orig_image = cv2.warpPerspective(orig_image, H, (500,500))

    #load the digit recognition model
    detector = load_model('digit_ml_model/results/digit_recognizer.h5')

    size = len(warp) // 9
    split = 1

    nums = []
    for i in range(0,9,split):
        row = []
        for j in range(0,9,split):
            num = warp[size*i:size*(i+split), size*j:size*(j+split)]
            try:
                digit = find_digit(num)
                x,y,w,h = cv2.boundingRect(digit) 
                #cv2.rectangle(orig_image, (x+size*j,y+size*i),
                #                        (x+w+size*j, y+h+size*i), (0,255,0), 3)
                num = num[y:y+h,x:x+w]
                if h-w > 0:
                    pad = (h-w)//2
                    num = cv2.copyMakeBorder(num, 0,0,pad,pad,cv2.BORDER_CONSTANT)
                else:
                    pad = (w-h)//2
                    num = cv2.copyMakeBorder(num, pad,pad, 0,0,cv2.BORDER_CONSTANT)
                num = cv2.copyMakeBorder(num, 10,10, 10,10,cv2.BORDER_CONSTANT)
                #cv2.imshow('test{}{}'.format(i,j), num)
                num = cv2.resize(num, (28,28))
                num_input = num.reshape((1,784)).astype('float32')
                prediction = detector.predict(num_input).tolist()[0]
                prediction = prediction.index(max(prediction))
                row.append(prediction)
                #print("({},{} -> {})".format(j+1, i+1, prediction))
            except Exception as e:
                print(e)
                row.append(0)
                pass
        nums.append(row)
    old_nums = deepcopy(nums)
    print(np.array(old_nums))
    success = solveSudoku(nums)
    print(np.array(nums))

    for i in range(9):
        for j in range(9):
            if old_nums[i][j] == 0: 
                cv2.putText(orig_image, str(nums[i][j]), (j*size+15, (i+1)*size-10),
                         font, 1.5, (0,0,255), 3, cv2.LINE_AA )
            else:
                cv2.putText(orig_image, str(old_nums[i][j]), (j*size+10, (i+1)*size-10),
                         font, 0.5, (0,0,255), 1, cv2.LINE_AA )


    cv2.imshow('warp', warp)
    cv2.imshow('orig', orig_image)
    cv2.waitKey(0)

if __name__=='__main__':
    main(sys.argv)

