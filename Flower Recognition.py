# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 11:01:26 2018

@author: Abdulrahman Alothman
"""


import numpy as np 
#diziler ve matrisler hızlı işlemler yapmamızı sağlayan Python kütüphanesidir
#import pandas as pd 
import matplotlib.pyplot as plt
#from PIL import  Image
#%matplotlib inline
plt.style.use('fivethirtyeight')


#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
#from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.contrib.keras import backend
from keras.optimizers import Adam
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils
#import keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix


import os
#flowers klasorun içindeki dosyaları listelemek 
print(os.listdir("flowers"))
#dirname() fonksiyonu, bir dosya yolunun dizin kısmını verir:
script_dir = os.path.dirname(".")
training_set_path = os.path.join(script_dir, 'flowers')
# 'dizin1\\dizin2\\dizin3\\flowers'
print(os.listdir("test_set"))
validation_set_path = os.path.join(script_dir, 'flowers')
test_set_path = os.path.join(script_dir, 'test_set')

classifier = Sequential()
input_size = (32, 32)
classifier.add(Conv2D(32, (5, 5), input_shape=(32,32,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides =  (2, 2)))

classifier.add(Conv2D(50, (5, 5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2),strides = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=500, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=250, activation='relu'))
classifier.add(Dense(units=5, activation='softmax'))


opt = Adam(lr=0.001)
classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#ağı öğretmek için her epochste datasetten  kaçar tane resim alacak 
batch_size = 10
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.10)
test_datagen = ImageDataGenerator(rescale=1./ 255)


training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 subset="training",
                                                 class_mode='categorical')
validation_set = validation_datagen.flow_from_directory(validation_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            subset="validation",
                                            class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            color_mode="rgb",
                                            shuffle = False,
                                            batch_size=1,
                                            class_mode='categorical')

print(training_set.class_indices)
print('You Have :',len(training_set.class_indices),'Class')



classifier.summary()
#model = load_model('my_model.h5')
classifier.load_weights('my_model_weights.h5')  ## trying to reload the compiled model
#print(classifier.get_weights())


#The model will not be trained on this data.(validation_data)
#Fitting the CNN to the images


'''model_info = classifier.fit_generator(training_set,
                         steps_per_epoch=3800//batch_size,
                         epochs=35,
                         validation_data=validation_set,
                         validation_steps=378//batch_size)
classifier.save_weights('my_model_weights.h5')'''

#model.save('my_model.h5')
scoreSeg = classifier.evaluate_generator(test_set,400,pickle_safe=False)
test_set.reset()
predict = classifier.predict_generator(test_set,400)



print('***tahmin Değerleri***')
print(np.argmax(predict, axis=1))
print('***Gerçek Değerleri***')
print(test_set.classes)

print(test_set.class_indices)

pred=np.argmax(predict, axis=1)

print("Confusion Matrix")
print(confusion_matrix(test_set.classes,pred))


print("Results")
print(classification_report(test_set.classes,pred,target_names=(sorted(test_set.class_indices.keys()))))

from sklearn.metrics import accuracy_score
print ('Accuracy:', accuracy_score(test_set.classes, np.argmax(predict, axis=1)))
'''

def plot_model_history(model_history):
    #print( range(1,len(model_history.history['acc'])+1))
    #print(model_history.history['acc'].pop())
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
plot_model_history(model_info)
'''

from IPython.display import Image, display

import os, random
img_locations = []
for d in os.listdir("flowers/"):
    directory = "flowers/" + d
    sample = [directory + '/' + s for s in random.sample(
        os.listdir(directory), int(random.random()*10))]
    img_locations += sample

def read_and_prep_images(img_paths, img_height=32, img_width=32):
    imgs = [load_img(img_path, target_size=(32, 32)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)

random.shuffle(img_locations)
imgs = read_and_prep_images(img_locations)
predictions = classifier.predict_classes(imgs)
classes = dict((v,k) for k,v in training_set.class_indices.items())

for img, prediction in zip(img_locations, predictions):
    display(Image(img))
    print(classes[prediction])
    

    






























