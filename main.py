import os
import BOW
import numpy as np
import pickle
import time

NO_OF_CLUSTERS=10
encoded={"personA":0,"personB":1,"personC":2,"personD":3,"personE":4}
# read the data
path="..\project\Dataset"
train_images = []
test_images = []
count = 0
for folder in os.listdir(path):
    for innerfolder in os.listdir(os.path.join(path, folder)):
        for file in os.listdir(os.path.join(path, folder,innerfolder)):
            if (file.__contains__(".png")):
                if(innerfolder.__contains__("Train")):
                    train_images.append(os.path.join(path, os.path.join(folder,innerfolder, file)))
                else:
                    test_images.append(os.path.join(path, os.path.join(folder,innerfolder, file)))

# train the model
# np.random.shuffle(train_images)
start_train = time.time()

kmeans,scale,svm,imgs_features=BOW.train_bow(train_images,encoded,NO_OF_CLUSTERS)
end_train = time.time()

svm_filename = 'svm_model.sav'
pickle.dump(svm, open(svm_filename, 'wb'))

kmeans_filename = 'kmeans_model.sav'
pickle.dump(kmeans, open(kmeans_filename, 'wb'))

scale_filename = 'scale_model.sav'
pickle.dump(scale, open(scale_filename, 'wb'))


# svm.save("svm.tfl")
# kmeans.save("kmeans.tfl")
# scale.save("svm.scale")
# test the model
start_test = time.time()
BOW.test_bow(test_images,encoded,kmeans,scale,svm,NO_OF_CLUSTERS)
end_test = time.time()

print(f"Training time: {end_train - start_train}s")
print(f"Testing time: {end_test - start_test}s")