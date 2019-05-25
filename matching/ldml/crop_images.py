import cv2
import os

img_folder = '/Users/ttnguyen/Documents/work/hades/vgg_face2/'

folders = os.listdir(img_folder + 'test/')
for folder in folders:
    images = os.listdir(img_folder + 'test/' + folder)
    for image in images:
        img = cv2.imread(img_folder + 'test/' + folder + '/' + image)
        h, w, d = img.shape
        side = w // 4
        top = h // 6
        bottom = h // 4
        img_cropped = img[top:h - bottom, side:w-side]
        if not os.path.exists(img_folder + 'test_crop/' + folder):
            os.makedirs(img_folder + 'test_crop/' + folder)
        cv2.imwrite(img_folder + 'test_crop/' + folder + '/' + image, img_cropped)
