
# -*- coding: utf-8 -*-

import os
from keras.applications.xception import Xception,preprocess_input

from keras.models import Sequential, Model,load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.pooling import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import glob


import numpy as np


def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))


img_width, img_height = 299, 299
nb_epoch = 10
batch_size = 20
result_dir = "../models/Xception/"

def getDataGen(trainDataNum):

  datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
  
  train_generator = datagen.flow_from_directory(
        "../data/train"+str(trainDataNum),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
  validation_generator = datagen.flow_from_directory(
        "../data/validation/",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
  return train_generator,validation_generator

def first_learn():
  #base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=None)
  input_tensor = Input(shape=(img_height, img_width, 3))
  base_model = Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)

  
  top_model = Sequential()
  top_model.add(Flatten(input_shape=base_model.output_shape[1:]))

  top_model.add(Dense(256, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1, activation='sigmoid'))
  model = Model(input=base_model.input, output=top_model(base_model.output))
  '''
  
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation = 'relu')(x)
  num_classes =1
  predictions = Dense(num_classes, activation = 'softmax')(x)
  model = Model(inputs = base_model.input, outputs = predictions)

  '''
  
  #108層までfreeze
  for layer in model.layers[:108]:
    layer.trainable = False

    # Batch Normalization の freeze解除
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True
    if layer.name.endswith('bn'):
        layer.trainable = True

  #109層以降、学習させる
  for layer in model.layers[108:]:
    layer.trainable = True


  #model.summary()
  model.compile(loss='binary_crossentropy',
              optimizer= Adam(),
              metrics=['accuracy'])

  #imagesList,labelsList = createTrainData(1)
  #validationImages,validationLabelsList = createValidationData()
  train_generator,validation_generator = getDataGen(1)

  history = model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        validation_data=validation_generator,
        verbose = 1)
  model.save(result_dir+"xception_dataAug_finetuning_train1.h5")
  save_history(history,result_dir+"history_dataAug_1")


def learn(numOfTrain, trainDataNum,loadModelNumOfTrain):

  model=load_model(result_dir+"xception_dataAug_finetuning_train"+str(loadModelNumOfTrain)+".h5")
  #108層までfreeze
  for layer in model.layers[:108]:
    layer.trainable = False

    # Batch Normalization の freeze解除
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True
    if layer.name.endswith('bn'):
        layer.trainable = True

  #109層以降、学習させる
  for layer in model.layers[108:]:
    layer.trainable = True


  #model.summary()
  train_generator,validation_generator = getDataGen(trainDataNum)

  history = model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        validation_data=validation_generator,
        verbose = 1)
  
  model.save(result_dir+"xception_dataAug_finetuning_train"+str(numOfTrain)+".h5")
  save_history(history,result_dir+"history_dataAug"+str(numOfTrain))


#first_learn()
learn(2,2,1)
learn(3,3,2)
learn(4,4,3)
learn(5,5,4)
