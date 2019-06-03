import numpy as np
import cv2
from imagezmq import imagezmq

# Create 2 different test images to send
# A green square on a black background
image1 = np.zeros((400, 400, 3), dtype='uint8')
green = (0, 255, 0)
cv2.rectangle(image1, (50, 50), (300, 300), green, 5)
# A red square on a black background
image2 = np.zeros((400, 400, 3), dtype='uint8')
red = (0, 0, 255)
cv2.rectangle(image2, (100, 100), (350, 350), red, 5)

sender = imagezmq.ImageSender(connect_to='tcp://192.168.0.1:5555')

image_window_name = 'From Sender'
while True:  # press Ctrl-C to stop image sending program
    sender.send_image(image_window_name, image1)
    sender.send_image(image_window_name, image2)
