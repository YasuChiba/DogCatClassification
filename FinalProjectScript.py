
import numpy as np
from PIL import Image
import glob
import os

import sys
import csv




img_width_xception_inception, img_height_xception_inception = 299, 299
img_width_vgg, img_height_vgg = 224, 224

if __name__ == '__main__':

    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input  
    from keras.applications.xception import Xception
    from keras.applications.xception import preprocess_input as xception_preprocess_input   
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
    from keras.models import Sequential, Model,load_model
    from keras.preprocessing import image   

 
    cat_img_name_list = glob.glob("./original_data/cat" + '/*.' + "jpg")
    dog_img_name_list = glob.glob("./original_data/dog" + '/*.' + "jpg")
    print("num of image  cat :  "+ str(len(cat_img_name_list)))   
    print("num of image  dog :  "+ str(len(dog_img_name_list)))   

    model_inception =load_model("./models/InceptionResNet_Finetuning/inceptionResNet_finetuning_train10.h5")
    model_xception =load_model("./models/Xception/xception_finetuning_train10.h5")
    model_vgg16 =load_model("./models/VGG_Finetuning/vgg16_finetuning_train10.h5")


    cat_resultArray = []  
    for img_name in cat_img_name_list:  
      img_vgg = image.load_img(img_name, target_size=(img_width_vgg, img_height_vgg))
      img_vgg = image.img_to_array(img_vgg)
      img_vgg = vgg16_preprocess_input(img_vgg)

      img_inception = image.load_img(img_name, target_size=(img_width_xception_inception, img_height_xception_inception))
      img_inception = image.img_to_array(img_inception)
      img_inception = inception_preprocess_input(img_inception)

      img_xception = image.load_img(img_name, target_size=(img_width_xception_inception, img_height_xception_inception))
      img_xception = image.img_to_array(img_xception)
      img_xception = xception_preprocess_input(img_xception)    
      predictedResult_vgg= model_vgg16.predict(np.array([img_vgg]))
      predictedResult_inception = model_inception.predict(np.array([img_inception]))
      predictedResult_xception = model_xception.predict(np.array([img_xception]))   
      sum_of_results = predictedResult_vgg[0][0] + predictedResult_inception[0][0] + predictedResult_xception[0][0]
      results_avg = sum_of_results/3

      filename = os.path.basename(img_name)
      filename = os.path.splitext(filename)[0]
      if results_avg >= 0.5:
          cat_resultArray.append(1)
          print("dog" + "  " + img_name)
      else:
          cat_resultArray.append(0)
          print("cat" + "  " + img_name)    
    

    dog_resultArray = []  
    for img_name in dog_img_name_list:  
      img_vgg = image.load_img(img_name, target_size=(img_width_vgg, img_height_vgg))
      img_vgg = image.img_to_array(img_vgg)
      img_vgg = vgg16_preprocess_input(img_vgg)

      img_inception = image.load_img(img_name, target_size=(img_width_xception_inception, img_height_xception_inception))
      img_inception = image.img_to_array(img_inception)
      img_inception = inception_preprocess_input(img_inception)

      img_xception = image.load_img(img_name, target_size=(img_width_xception_inception, img_height_xception_inception))
      img_xception = image.img_to_array(img_xception)
      img_xception = xception_preprocess_input(img_xception)    
      predictedResult_vgg= model_vgg16.predict(np.array([img_vgg]))
      predictedResult_inception = model_inception.predict(np.array([img_inception]))
      predictedResult_xception = model_xception.predict(np.array([img_xception]))   
      sum_of_results = predictedResult_vgg[0][0] + predictedResult_inception[0][0] + predictedResult_xception[0][0]
      results_avg = sum_of_results/3

      filename = os.path.basename(img_name)
      filename = os.path.splitext(filename)[0]
      if results_avg >= 0.5:
          dog_resultArray.append(1)
          print("dog" + "  " + img_name)
      else:
          dog_resultArray.append(0)
          print("cat" + "  " + img_name)
        
    print(len(cat_resultArray))
    print(len(dog_resultArray))

    cat_fail_num = cat_resultArray.count(1)
    dog_fail_num = dog_resultArray.count(0)

    print(cat_fail_num)
    print(dog_fail_num)
    print("accuracy : ", (cat_fail_num + dog_fail_num)/(len(cat_resultArray) + len(dog_resultArray)))


