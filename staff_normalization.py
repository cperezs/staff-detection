import numpy as np
from staff.normalization import *
import cv2

def staff_normalization(image):
    img_lines = isolate_lines(img)
    paths = compute_paths(img_lines)
    normalized = normalize(img, paths)
    cropped = normalization_five_peaks(normalized)    
    return cropped
    
    

img = cv2.imread('corpus/9.jpg')
normalized = staff_normalization(img)
cv2.imwrite('prueba.png',normalized)

