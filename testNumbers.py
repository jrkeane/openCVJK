
import cv2
import numpy as np
import os
from os import path
import labelNumbersInit as lni


frame = cv2.imread('/media/kookaburra/JimsDisk/Ninox/numbers/Ninox4720_0_Six.jpg')
boundimg = cv2.rectangle(frame, (1, 1), (12, 16), (255, 255,255), 1)
cv2.imshow("Detection", frame)

k = cv2.waitKey(0) & 0xff
