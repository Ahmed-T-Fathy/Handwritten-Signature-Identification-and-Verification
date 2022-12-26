import cv2

import model_healpers
import numpy as np
import os

IMG_SIZE=50
path=".\Dataset\personB"

encoded={"forged":np.array([0,1]),"real":np.array([1,0])}


model=model_healpers.create_model(IMG_SIZE)




if (os.path.exists('modelB.tfl.meta')):
    model.load('./modelB.tfl')
else:
     X_train, Y_train, X_test, Y_test = model_healpers.get_data(path, IMG_SIZE, encoded)
     model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=25,
               show_metric=True, run_id="stage 2 personB")
     model.save('modelB.tfl')




# model.save('modelB.tfl')

def get_accuarcy():
     # test = model.predict(X_test)
     print('==================================================================')
     train_score = model.evaluate(X_train, Y_train)
     test_score = model.evaluate(X_test, Y_test)
     print("Model B",)
     print('train accuarcy:',train_score[0]*100,'%')
     print('test accuarcy:',test_score[0]*100,'%')
     print('==================================================================')

# get_accuarcy()
# img=cv2.imread("./real.png",0)
def test_img(img):
     img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
     img=np.array(img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
     type = model.predict_label(img)
     type=type[0][0]
     if(type==1):
          return "forged"
     else:
          return "real"

# print(test_img(img))