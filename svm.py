import time
import numpy as np
import os
import cv2.cv2 as cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

train_dir = '../datasets-fer2013/train/'
test_dir = '../datasets-fer2013/test/'


#load data sets

def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            # image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    label_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    # print(label_dict)
    y = [label_dict[class_name[i]] for i in range(len(class_name))]
    return img_data_array, y


# {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
X_train, y_train = create_dataset(train_dir)
y_train = np.array(y_train)
print('training set loaded')

# now X is a list, every element is a ndarray (48*48)
X_test, y_test = create_dataset(test_dir)
y_test = np.array(y_test)
print('test set loaded')


# extract hog features for X_train
hog_features_train = []
for image in X_train:
    features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    hog_features_train.append(features)
hog_features_train = np.array(hog_features_train)
print('hog feature of training set obtained')

# extract hog features for X_test
hog_features_test = []
for image in X_test:
    features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    hog_features_test.append(features)
hog_features_test = np.array(hog_features_test)
print('hog feature of test set obtained')


#train
print("Fitting the classifier to the training set")
start_time = time.time()

random_state = 0

# hyperparameter search
param_grid = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}

clf = RandomizedSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10)
clf = clf.fit(hog_features_train, y_train)
pickle.dump(clf, open('trained_model.sav', 'wb'))


training_time = time.time() - start_time
print("training finished in {0:.1f} sec".format(training_time))
print("Best estimator found by random search:", clf.best_estimator_)


accuracy_train = clf.score(hog_features_train, y_train)
print('train accuracy: {:.2%}'.format(accuracy_train))


#test
clf = pickle.load(open('trained_model.sav', 'rb'))
y_pred = clf.predict(hog_features_test)

print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
disp.plot()
plt.show()

