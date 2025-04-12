import numpy as np
import cv2

highThresh = 0.4
lowThresh = 0.1
imgFileLst = ['./villageRoad.jpg']  # Ensure the path is correct

def sobel(img):
    '''
    Detects edges using sobel kernel
    '''
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)  # detects horizontal edges
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)  # detects vertical edges
    # combine both edges
    return cv2.bitwise_or(opImgx, opImgy)  # does a bitwise OR of pixel values at each pixel

def sketch(frame):
    # Blur it to remove noise
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    # make a negative image
    invImg = 255 - frame

    # Detect edges from the input image and its negative
    edgImg0 = sobel(frame)
    edgImg1 = sobel(invImg)
    edgImg = cv2.addWeighted(edgImg0, 1, edgImg1, 1, 0)  # different weights can be tried too

    # Invert the image back
    opImg = 255 - edgImg
    return opImg

if __name__ == '__main__':
    for imgFile in imgFileLst:
        print(f"Processing: {imgFile}")
        img = cv2.imread(imgFile, 0)  # Read the image in grayscale
        if img is None:
            print(f"Error: Could not read image file '{imgFile}'. Please check the file path.")
            continue
        opImg = sketch(img)
        cv2.imshow(imgFile, opImg)

    cv2.waitKey()
    cv2.destroyAllWindows()
