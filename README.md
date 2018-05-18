# German-Traffic-Signs-Detector
Kiwi Campus Deep Learning Challenge.

This project contains 3 different models performing  a classification task on data from the http://benchmark.ini.rub.de/Dataset_GTSDB/. The first model is a logistic regression from sklearn, the second one is a implementation of a logistic regression with multiclass output and the third one is a LeNet-5 architecture CNN.

### Prerequisites

You should create a new conda environment from the environment.yml file.

```
conda env create -f environment.yml
```

Then activate this new environment named kiwichallenge.

To download the data run: 

```
python app.py download
```

This may take a while but if you just want to test (not infer) the model with your own data, just save the images specifying the real class in the name of the image as the first two characters. e.g 14stopsign.png. You do not have to save the images in this way if you just want to run inference in the images. If you are going to use your own images for training and testing please follow the convention mentioned above. 00 for class 0 01 for class 1 and so on.

To train the models run: 

```
python app.py train -m [model-name] -d [path]
```

model-name should be: model1, model2 or model3.
path should be the path from the root directory of the proyect to where the data is stored. e.g: images/train

To test the models run: 

```
python app.py test -m [model-name] -d [path]
```

model-name should be: model1, model2 or model3.
path should be the path from the root directory of the proyect to where the data is stored. e.g: images/test

To infer with the models run: 

```
python app.py infer -m [model-name] -d [path]
```

model-name should be: model1, model2 or model3.
path should be the path from the root directory of the proyect to where the data is stored. e.g: images/user

## Authors

* **Juan Daniel Morales**