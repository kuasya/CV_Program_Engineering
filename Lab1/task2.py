import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


# Реализация Хромакея    
def processImgs(img1,img2):
  # Приводим изображения к одному размеру (основное - первое)
  img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
  
  # Преобразуем изображение в HSV
  hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
  green = np.uint8([[[0, 255, 0]]])
  hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
  h = hsv_green[0][0][0]

  # Создание маски для заданного цвета
  # (35,50,50) ~ (85,255,255)
  lower_color = np.array([h-25, 50, 50], dtype = "uint8")
  upper_color = np.array([h+25, 255, 255], dtype = "uint8")
  # Создание маски для заданного цвета
  img_mask = cv2.inRange(hsv, lower_color, upper_color)

  # Размытие краев маски
  img_mask = cv2.medianBlur(img_mask,7)

  # Создание фонового изображения с маской
  img_bgMasked = cv2.bitwise_and(img2, img2, mask = img_mask) 

  # Убираем цвет из оригинального изображения
  img_invMask = cv2.bitwise_not(img_mask)
  img_masked = cv2.bitwise_and(img1, img1, mask = img_invMask) 

  # Объединяем изображения
  img_output = cv2.bitwise_or(img_bgMasked, img_masked) 
  return img_output


# указываем путь к изображению
# Загрузка изображениий
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Применяем хромакей
result_image = chroma_key('image1.jpg', 'image2.jpg')
cv2.imshow('Chroma Key Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
