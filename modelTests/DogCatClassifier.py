
import numpy as np
from PIL import Image
import glob


import sys
import csv




img_width_xception_inception, img_height_xception_inception = 299, 299
img_width_vgg, img_height_vgg = 224, 224

#args[1]: 画像の入ったフォルダ  
#args[2]: outputのファイル名
if __name__ == '__main__':
    args = sys.argv
    if len(args) <= 2:
      print("argument 1: test pictures folder    2: output file name")
      sys.exit(1)   

    from keras.applications.inception_v3 import InceptionV3,preprocess_input
    from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input  
    from keras.applications.xception import Xception,preprocess_input
    from keras.applications.xception import preprocess_input as xception_preprocess_input   
    from keras.applications.vgg16 import VGG16, preprocess_input
    from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
    from keras.models import Sequential, Model,load_model
    from keras.preprocessing import image   

    image_floder_name = args[1]
    output_file_name = args[2]  
    img_name_list = glob.glob(image_floder_name + '/*.' + "jpg") 
    print("num of image:  "+ str(len(img_name_list)))   
    if len(img_name_list) == 0:
      sys.exit(1)   

    model_inception =load_model("../models/InceptionResNet_Finetuning/inceptionResNet_finetuning_train10.h5")
    model_xception =load_model("../models/Xception/xception_finetuning_train10.h5")
    model_vgg16 =load_model("../models/VGG_Finetuning/vgg16_finetuning_train10.h5")


    resultArray = []  
    index = 1
    for img_name in img_name_list:  
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

      if results_avg >= 0.5:
          resultArray.append([index,1])
          print("dog" + "  " + img_name)
      else:
          resultArray.append([index,0])
          print("cat" + "  " + img_name)    
      index += 1
    

    with open(output_file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(resultArray)
  
