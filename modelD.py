import os
import time

import cv2

import model_healpers
import numpy as np

IMG_SIZE=50
path=".\Dataset\personD"

encoded={"forged":np.array([0,1]),"real":np.array([1,0])}


model=model_healpers.create_model(IMG_SIZE)




if (os.path.exists('modelD.tfl.meta')):
    model.load('./modelD.tfl')
else:
     X_train, Y_train, X_test, Y_test = model_healpers.get_data(path, IMG_SIZE, encoded)
     start_train = time.time()
     model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=30,
               show_metric=True, run_id="stage 2 personD")
     end_train = time.time()
     model.save('modelD.tfl')
     print('==================================================================')
     print("Model D", )
     print(f"Training time: {end_train - start_train}s")



# model.save('modelB.tfl')

def get_accuarcy():
     # test = model.predict(X_test)

     train_score = model.evaluate(X_train, Y_train)
     start_test = time.time()
     test_score = model.evaluate(X_test, Y_test)
     end_test = time.time()
     print(f"Testing time: {end_test - start_test}s")
     print('train accuarcy:',train_score[0]*100,'%')
     print('test accuarcy:',test_score[0]*100,'%')
     print('==================================================================')

get_accuarcy()
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