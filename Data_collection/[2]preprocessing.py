import numpy as np
import cv2, os, glob
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontface.xml')

data_path = 'str'
save_path = 'str2'

if not os.path.exists(save_path):
    os.mkdir(save_path)

def data_preprocessing():
    images = [cv2.imread(file) for file in glob.glob(data_path+"/*.jpg")]
    cnt = 0
    for image in images:
        faces = face_cascade.detectMultiScale(image, 1.03, 5)
        for (x,y,w,h) in faces:
            cnt += 1
            cropped_img = image[y:y+h, x:x+w]
            output_img = ''
            if cropped_img.shape[0] >= 150:
                output_img = cv2.resize(cropped_img, dsize=(150, 150), interpolation=cv2.INTER_AREA)
            else:
                output_img = cv2.resize(cropped_img, dsize=(150, 150), interpolation=cv2.INTER_LINEAR)
            
            cv2.imwrite(save_path+'/'+str(cnt)+'.jpg', output_img)

def main():
    data_preprocessing()
    print('end')

if __name__ == '__main__':
    main()