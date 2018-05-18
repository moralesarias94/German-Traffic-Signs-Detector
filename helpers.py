from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from shutil import copyfile
from sklearn.preprocessing import OneHotEncoder
import requests, zipfile, io, os
from sklearn.externals import joblib
from sklearn.utils import shuffle

path_to_root = os.path.dirname(os.path.abspath(__file__))
#Map for displaying class names.
infer_map = {
0 : 'speed limit 20 (prohibitory)',
1 : 'speed limit 30 (prohibitory)',
2 : 'speed limit 50 (prohibitory)',
3 : 'speed limit 60 (prohibitory)',
4 : 'speed limit 70 (prohibitory)',
5 : 'speed limit 80 (prohibitory)',
6 : 'restriction ends 80 (other)',
7 : 'speed limit 100 (prohibitory)',
8 : 'speed limit 120 (prohibitory)',
9 : 'no overtaking (prohibitory)',
10 : 'no overtaking (trucks) (prohibitory)',
11 : 'priority at next intersection (danger)',
12 : 'priority road (other)',
13 : 'give way (other)',
14 : 'stop (other)',
15 : 'no traffic both ways (prohibitory)',
16 : 'no trucks (prohibitory)',
17 : 'no entry (other)',
18 : 'danger (danger)',
19 : 'bend left (danger)',
20 : 'bend right (danger)',
21 : 'bend (danger)',
22 : 'uneven road (danger)',
23 : 'slippery road (danger)',
24 : 'road narrows (danger)',
25 : 'construction (danger)',
26 : 'traffic signal (danger)',
27 : 'pedestrian crossing (danger)',
28 : 'school crossing (danger)',
29 : 'cycles crossing (danger)',
30 : 'snow (danger)',
31 : 'animals (danger)',
32 : 'restriction ends (other)',
33 : 'go right (mandatory)',
34 : 'go left (mandatory)',
35 : 'go straight (mandatory)',
36 : 'go right or straight (mandatory)',
37 : 'go left or straight (mandatory)',
38 : 'keep right (mandatory)',
39 : 'keep left (mandatory)',
40 : 'roundabout (mandatory)',
41 : 'restriction ends (overtaking) (other)',
42 : 'restriction ends (overtaking (trucks)) (other)'
}


#Helper functions
def read_transform_all_images(path, infer=False):
    """
    Read the images, transform them into a list of keypoint descriptors. Returns a list with a row per image with
    it's descriptors. Also returns a list with the label of each row and the image names. 
    """
    path = os.path.join(path_to_root, path)
    img_names = os.listdir(path)
    transformed_images = []
    img_classes = []
    transformed_image_names = []
    count_images = 0
    sift = cv2.xfeatures2d.SIFT_create()
    for img_name in img_names:
        if (not img_name.startswith('.') and os.path.isfile(os.path.join(path, img_name))):
            count_images += 1
            img_path = os.path.join(path, img_name)
            img_descriptors = transform_image(img_path, sift)
            #Ignore image if no keypoints detected
            if(len(img_descriptors)>0):
                transformed_images.append(img_descriptors)
                img_classes.append(get_class_from_name(img_name, infer))
                transformed_image_names.append(img_name)
    return transformed_images, np.array(img_classes), transformed_image_names
    
def get_class_from_name(img_name, infer=False):
    """Returns the class of the image"""
    try:
        img_class = int(img_name[:2])
    except:
        img_class = None
        if(not infer):
            print("Please provide the #class in the name of the file as the first characters.")
    return img_class

def transform_image(img_path, sift):
    """
    Loads the image, convert it to grayscale, extract the features.
    """
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    max_width_height = 32
    img = cv2.resize(img, (max_width_height, max_width_height))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    if(len(kp) == 0):
        des = []
    return des
    
def create_cluster(images, k):
    """Creates the cluster to perform dimensionality reduction."""
    kmeans = KMeans(n_clusters=k, random_state=0).fit(images)
    return kmeans

def create_samples(all_images_features, clusters):
    """Creates samples from image features using the clusters."""
    all_features = []
    for image in all_images_features:
        all_features.append(encode_features(image, clusters))
    return all_features

def encode_features(image_features, clusters):
    """Get the image features and creates a normalized histogram with the features."""
    histogram = np.array([0 for center in clusters.cluster_centers_])
    preds = clusters.predict(image_features)
    for pred in preds:
        histogram[pred] += 1
    histogram = (histogram-min(histogram))/(max(histogram)-min(histogram))
    return histogram

def get_train_samples(train_features, kmeans, model='model1'):
    """Calculates the idf of the features and returns the samples created multiplied with the idf."""
    train_samples = pd.DataFrame(create_samples(train_features, kmeans))
    idf = pd.DataFrame(train_samples.apply(lambda x: np.log(len(train_samples)/len(x[x>0])), axis=0).values, columns=['idf'])
    idf.to_csv(os.path.join(path_to_root, 'models/aux/idf%s.txt' % model), index=False)
    train_samples = train_samples * idf['idf']
    return train_samples

def get_test_samples(test_features, kmeans, model='model1'):
    """Loads the idf of the features and returns the samples created multiplied with the idf."""
    test_samples = pd.DataFrame(create_samples(test_features, kmeans))
    idf = pd.read_csv(os.path.join(path_to_root, 'models/aux/idf%s.txt' % model), header=0)
    if(len(idf) == len(kmeans.cluster_centers_)):
        test_samples = test_samples * idf['idf']
    else:
        print("There is a problem with the amount of features (#clusters in kmeans model) and idf lenght. Run train again.")
    return test_samples

def show_predictions(preds, d, img_names):
    """Creates a window for each image and the predicted label for them."""
    for pred, img_name in zip(preds, img_names):
        img_path = os.path.join(os.path.join(path_to_root,d), img_name)
        img = cv2.imread(img_path)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Class: %s' % infer_map[pred])
        plt.show()

def random_batch(X_train, y_train, batch_size):
    """Returns a random batch of size batch_size."""
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

def get_images(path, infer=False):
    """Get BGR images with its classes and the names of them. Also resize all images to 28x28x3."""
    path = os.path.join(path_to_root, path)
    img_names = os.listdir(path)
    images = []
    img_classes = []
    transformed_image_names = []
    count_images = 0
    for img_name in img_names:
        if (not (img_name.startswith('.')) and os.path.isfile(os.path.join(path, img_name))):
            count_images += 1
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            height, width, channels = img.shape
            max_width_height = 28
            img = cv2.resize(img, (max_width_height, max_width_height))
            img = standarize_img(img)
            images.append(img)
            img_classes.append(get_class_from_name(img_name, infer))
            transformed_image_names.append(img_name)
    return np.array(images, 'float32'), np.array(img_classes, 'float32'), transformed_image_names

def transform_input(images, labels, infer=False):
    """Zero pad the images to get a shape of 32x32x3"""
    images = np.pad(images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    if(not infer):
        enc = OneHotEncoder(43)
        try:
            labels = enc.fit_transform(labels.reshape(-1, 1)).todense()
        except:
            print('Can not train with infer data.')
    return images, labels

def standarize_img(img):
    mean = np.mean(img)
    dev = np.std(img)
    return (img - mean) / dev