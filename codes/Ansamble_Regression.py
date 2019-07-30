from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input

from keras.applications.xception import Xception,preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input


from keras.models import Sequential, Model,load_model
from keras.preprocessing import image

from sklearn import linear_model
from sklearn.externals import joblib

import numpy as np
from PIL import Image
import glob


model_inception =load_model("../models/InceptionResNet_Finetuning/inceptionResNet_finetuning_train10.h5")
model_xception =load_model("../models/Xception/xception_finetuning_train10.h5")
model_vgg16 =load_model("../models/VGG_Finetuning/vgg16_finetuning_train10.h5")

img_width_xception_inception, img_height_xception_inception = 299, 299
img_width_vgg, img_height_vgg = 224, 224

def loadAllImage(img_name_list, preprocess_input,img_width,img_height):

  imageList = []
  for imgName in img_name_list:
    img = image.load_img(imgName, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    imageList.append(x)
  return np.asarray(imageList)


def predict(cat_images, dog_images, model):
  result = []
  for tmpImage in cat_images:
      predictedResult= model.predict(np.array([tmpImage]))
      result.append(float(predictedResult[0][0]))
      print("cat   ", float(predictedResult[0][0]))

  for tmpImage in dog_images:
      predictedResult= model.predict(np.array([tmpImage]))
      result.append(float(predictedResult[0][0]))
      print("dog   ", float(predictedResult[0][0]))

  return np.array(result)


def predict_from_models(cat_image_dir,dog_image_dir):
  dog_img_name_list = glob.glob(dog_image_dir + '/*.' + "jpg")
  cat_img_name_list = glob.glob(cat_image_dir + '/*.' + "jpg")

  inception_dog_images = loadAllImage(dog_img_name_list, inception_preprocess_input,img_width_xception_inception,img_height_xception_inception)
  inception_cat_images = loadAllImage(cat_img_name_list, inception_preprocess_input,img_width_xception_inception,img_height_xception_inception)

  xception_dog_images = loadAllImage(dog_img_name_list, xception_preprocess_input,img_width_xception_inception,img_height_xception_inception)
  xception_cat_images = loadAllImage(cat_img_name_list, xception_preprocess_input,img_width_xception_inception,img_height_xception_inception)

  vgg_dog_images = loadAllImage(dog_img_name_list, vgg16_preprocess_input,img_width_vgg,img_height_vgg)
  vgg_cat_images = loadAllImage(cat_img_name_list, vgg16_preprocess_input,img_width_vgg,img_height_vgg)

  catLabels =  np.array([[0] for i in range(0,len(cat_img_name_list))])
  dogLabels =  np.array([[1] for i in range(0,len(dog_img_name_list))])

  inception_result = predict(inception_cat_images, inception_dog_images, model_inception)
  xception_result = predict(xception_cat_images, xception_dog_images, model_xception)
  vgg_result = predict(vgg_cat_images, vgg_dog_images, model_vgg16)

  labelList = []
  for tmp in catLabels:
      labelList.append(0)
  for tmp in dogLabels:
      labelList.append(1)
  labelList = np.array(labelList)

  X = []
  for index in range(0, len(labelList)):
    X.append(np.array([inception_result[index],xception_result[index],vgg_result[index]]))
  X = np.array(X)

  return X, labelList

def first_learn():
  X, labelList = predict_from_models("../data/train1/cats","../data/train1/dogs")
  clf = linear_model.LinearRegression()
  clf.fit(X, labelList)
  score = clf.score(X, labelList)
  print(score)
  joblib.dump(clf, '../models/Ansamble/regression1.pkl')
  with open('../models/Ansamble/history.txt', 'a') as f:
    print(score, file=f)


def learn(numOfTrain, trainDataNum,loadModelNumOfTrain):
  X, labelList = predict_from_models("../data/train"+str(trainDataNum)+"/cats","../data/train"+str(trainDataNum)+"/dogs")
  clf = joblib.load("../models/Ansamble/regression"+str(loadModelNumOfTrain)+".pkl")
  clf.fit(X, labelList)
  print(clf.score(X, labelList))
  joblib.dump(clf, "../models/Ansamble/regression"+str(numOfTrain)+".pkl")
  with open('../models/Ansamble/history.txt', 'a') as f:
    print(score, file=f)

first_learn()
learn(2,2,1)
learn(3,3,2)
learn(4,4,3)
learn(5,5,4)
