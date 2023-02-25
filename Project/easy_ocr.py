import easyocr
import cv2

img_path = './using_image/2.jpg'

a = cv2.imread(img_path)
reader = easyocr.Reader(['en'])

result_detect = reader.detect(a)
result_read = reader.readtext(a)
result_reco = reader.recognize(a)

print(len(result_detect))

print(len(result_read))

print(result_reco)

