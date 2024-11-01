import cv2
import numpy as np
from keras.models import load_model
from keras.datasets import cifar10
import glob
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
#(x_train,y_train),(x_test,y_test)=cifar10.load_data() 
#print(x_test)
actual = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0] 
c=0
d=0
predicted=[]
def draw_test(pred):
    global c
    global d
    if pred == "[0]":
        c=c+1
        pred = 1
    if pred == "[1]":
        pred = 0
        d=d+1
    return pred

classifier = load_model('pred.h5')
for filename in listdir('N1_output'):

    a=cv2.imread('N1_output/'+filename)
    a=a.reshape(1,64,64,3)
    print(a.shape)
    

    
    
    ## Get Prediction
    res = str(classifier.predict_classes(a, 1, verbose = 0)[0])
    predicted.append(draw_test(res))
    print(res)    
    #x_test.append(a)


results = confusion_matrix(actual, predicted)
print(classification_report(actual, predicted))
print(accuracy_score(actual, predicted))
 