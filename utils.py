import cv2
import torch
import numpy as np

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert colors (MNIST style)
    img = cv2.bitwise_not(img)

    # threshold to isolate digit
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # find contours (digit boundary)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        img = img[y:y+h, x:x+w]

    # resize digit while preserving shape
    img = cv2.resize(img, (20,20))

    # place digit in 28x28 canvas
    canvas = np.zeros((28,28))

    canvas[4:24,4:24] = img

    img = canvas / 255.0

    img = img.reshape(1,1,28,28)

    img = torch.tensor(img, dtype=torch.float32)

    return img