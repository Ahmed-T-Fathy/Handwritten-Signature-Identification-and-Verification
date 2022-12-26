import cv2
import BOW
import pickle
import tflearn as tf

img =cv2.imread("./Dataset/personD/Test/personD_14.png",0)

svm=pickle.load(open("svm_model.sav", 'rb'))
kmeans=pickle.load(open("kmeans_model.sav", 'rb'))
scale=pickle.load(open("scale_model.sav", 'rb'))

person=BOW.test_script(img,svm,kmeans,scale,10)

print("***********************************")
print(person)
print("***********************************")

if (person=="personA"):
    import modelA
    type=modelA.test_img(img)
elif(person=="personB"):
    import modelB
    type = modelB.test_img(img)
elif(person=="personC"):
    import modelC
    type = modelC.test_img(img)
elif(person=="personD"):
    import modelD
    type = modelD.test_img(img)
elif(person=="personE"):
    import modelE
    type = modelE.test_img(img)



print("the signature belong to -->",person,"<-- and it is -->",type,"<-- signature.")