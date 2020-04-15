import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from skimage import color,transform
import matplotlib.image as mpimg
from skimage.feature import hog

X = []
Y = []

dataset_size = 12500
datasetpath = "dataset/train/"
for i in range(dataset_size):
    img_name  = datasetpath+"cat." + str(i) + '.jpg'
    img_cat = color.rgb2gray(mpimg.imread(img_name))
    img = transform.resize(img_cat, (128,64) , mode='constant')
    fcat, hog_img = hog(img, orientations = 9, pixels_per_cell = (8,8),cells_per_block = (2,2), visualise = True , block_norm='L2-Hys')
    X.append(fcat)
    Y.append(0)

for i in range(dataset_size):
    img_name  =  datasetpath +"dog." + str(i) + '.jpg'
    img_dog = color.rgb2gray(mpimg.imread(img_name))
    img = transform.resize(img_dog, (128, 64) , mode='constant')
    fdog, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True , block_norm='L2-Hys')
    X.append(fdog)
    Y.append(1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)


svc = svm.SVC(kernel='poly', C=1 , degree=1).fit(X_train, y_train)
#lin_svc = svm.LinearSVC(C=1).fit(X, Y)


predictions = svc.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(str(accuracy*100) + ' % , Dataset Size : ' + str(dataset_size  ))