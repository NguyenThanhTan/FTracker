import cv2

bin = 16

def get_img_hist(img):
    h = cv2.calcHist([img], [0, 1, 2], None, [bin, bin, bin],
                     [0, 256, 0, 256, 0, 256])
    return h