import cv2
import numpy as np

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train_bow(train_images,encoded,NO_OF_CLUSTERS):
    NO_OF_CLUSTERS=NO_OF_CLUSTERS
    sift = cv2.xfeatures2d.SIFT_create()
    descriptors = []
    trainingLabels = []

    for image in train_images:

        label = 0
        if image.__contains__("personA"):
            label = encoded["personA"]
        elif image.__contains__("personB"):
            label = encoded["personB"]
        elif image.__contains__("personC"):
            label = encoded["personC"]
        elif image.__contains__("personD"):
            label = encoded["personD"]
        else:
            label = encoded["personE"]
        img = cv2.imread(image, 0)

        # find the keypoints an[pllld descriptors with SIFT
        kp1, des = sift.detectAndCompute(img, None)
        descriptors.append(des)
        trainingLabels.append(label)

    # flatten descriptors of whole images
    descriptor_list = np.array(descriptors[0])
    for descriptor in (descriptors[1:]):
        descriptor_list = np.vstack((descriptor_list, descriptor))

    # clustring the descriptors
    kmeans = KMeans(n_clusters=NO_OF_CLUSTERS).fit(descriptor_list)

    # feature extraction
    mega_histogram = np.array([np.zeros(NO_OF_CLUSTERS) for i in range(len(train_images))])
    # old_count = 0
    for i in range(len(train_images)):
        l = len(descriptors[i])
        for j in range(l):
            feature = descriptors[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            mega_histogram[i][idx] += 1

    scale = StandardScaler().fit(mega_histogram)
    imgs_features = scale.transform(mega_histogram)

    x_scalar = np.arange(NO_OF_CLUSTERS)
    y_scalar = np.array([abs(np.sum(imgs_features[:, h], dtype=np.int32)) for h in range(NO_OF_CLUSTERS)])

    #
    #
    # plt.bar(x_scalar, y_scalar)
    # plt.xlabel("Visual Word Index")
    # plt.ylabel("Frequency")
    # plt.title("Complete Vocabulary Generated")
    # plt.xticks(x_scalar + 0.4, x_scalar)
    # plt.show()

    svm = SVC().fit(imgs_features, trainingLabels)
    return kmeans,scale,svm,imgs_features

def test_bow(test_images,encoded,kmeans,scale,svm,NO_OF_CLUSTERS):
    sift = cv2.xfeatures2d.SIFT_create()
    descriptors = []
    testingLabels = []

    for image in test_images:

        label = 0
        if image.__contains__("personA"):
            label = encoded["personA"]
        elif image.__contains__("personB"):
            label = encoded["personB"]
        elif image.__contains__("personC"):
            label = encoded["personC"]
        elif image.__contains__("personD"):
            label = encoded["personD"]
        else:
            label = encoded["personE"]
        img = cv2.imread(image, 0)

        # find the keypoints an[pllld descriptors with SIFT
        kp1, des = sift.detectAndCompute(img, None)
        descriptors.append(des)
        testingLabels.append(label)

    descriptor_list = np.array(descriptors[0])
    for descriptor in (descriptors[1:]):
        descriptor_list = np.vstack((descriptor_list, descriptor))

    mega_histogram = np.array([np.zeros(NO_OF_CLUSTERS) for i in range(len(test_images))])
    # old_count = 0
    for i in range(len(test_images)):
        l = len(descriptors[i])
        for j in range(l):
            feature = descriptors[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            mega_histogram[i][idx] += 1

    imgs_features = scale.transform(mega_histogram)
    prediction=svm.predict(imgs_features)
    accuracy=accuracy_score(testingLabels,prediction)

    print("BOW testing accuracy:",accuracy*100,'%')


def test_script(img,svm,kmeans,scale,NO_OF_CLUSTERS):
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des=sift.detectAndCompute(img, None)

    img_feature=np.array(np.zeros(NO_OF_CLUSTERS))
    l = len(des)
    for j in range(l):
        feature = des[j]
        feature = feature.reshape(1, 128)
        idx = kmeans.predict(feature)
        img_feature[idx] += 1


    # img_feature=np.array(img_feature).reshape(-1, 1)
    imgs=[]
    for i in range(2):
        imgs.append(img_feature)
    imgs=scale.transform(imgs)

    prediction = svm.predict(imgs)
    if(prediction[0]==0):
        return "personA"
    elif(prediction[0]==1):
        return "personB"
    elif (prediction[0] == 2):
        return "personC"
    elif (prediction[0] == 3):
        return "personD"
    else:
        return "personE"