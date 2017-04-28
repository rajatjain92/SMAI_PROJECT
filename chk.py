import data
import cv2 as cv

cv.namedWindow('x')


a = data.get_dir('B_hair')
b = data.read_dir(a, True)

im = b['B_hair'][0][1]
