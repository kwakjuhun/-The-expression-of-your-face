import cv2 
from keras.models import load_model
import numpy as np

model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('Data_collection/haarcascade_frontface.xml')


def facial_expression_detection():
    cap = cv2.VideoCapture(0) 
    cap.set(3, 720) 
    cap.set(4, 1080) 

    while True: 
        ret, frame = cap.read() 
        faces = face_cascade.detectMultiScale(frame, 1.3,5)
        for(x,y,w,h) in faces:
            image = frame[y:y+h, x:x+w]
            dst = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dst = cv2.resize(dst, dsize=(150, 150))
            dst = np.array(dst)
            dst = dst.reshape(-1, 150, 150, 1)
            dst = dst/2

            predict = model.predict(dst)
            label = np.argmax(predict[0])
            check = ['SOSO','HAPPY']
            cv2.putText(frame, check[label], (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))        

        cv2.imshow('test', frame) 
        k = cv2.waitKey(1) 
        if k == 27: 
            break 
    cap.release() 
    cv2.destroyAllWindows()


def main():    
    facial_expression_detection()
    print('end')

if __name__ == '__main__':
    main()