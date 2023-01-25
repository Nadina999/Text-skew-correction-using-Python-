import cv2
import numpy as np
import math


img = cv2.imread('/home/nadinaa77/Desktop/scanned.jpg',0)


def getAngle(a, b, c):
  ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
  return ang + 360 if ang < 0 else ang

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],flags=cv2.INTER_LINEAR)
  return result

kernel_size = 5

blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

low_threshold = 50

high_threshold = 150

edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1

theta = np.pi / 180

threshold = 15

min_line_length = 50

max_line_gap = 20

line_image = np.copy(img) * 0

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

for line in lines:
  for x1,y1,x2,y2 in line:
   C=[x1, y1]
   A=[x2, y2]
   B=[x2, y1]
   angle=getAngle(B,C,A)
   cv2.putText(img, f"Angle: {angle}", (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 1)
   break
  break

output=rotate_image(img, angle)
cv2.imshow('output', output)
cv2.imshow('input', img)
filename='rotated.jpg'
cv2.imwrite(filename, output)
cv2.waitKey(0)
cv2.destroyAllWindows()