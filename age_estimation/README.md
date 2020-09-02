# Lunar impact craters ages estimation  

# Requirements:
Tested on 
* python 3.6
* pytorch 0.4.0
* pretrainedmodels 0.7.4

# Data preparing
## image data preparing
For image data, the DOM images (gray images) of the aged craters and the identified craters are resized to 256*256.

## attribute data preparing
For attribute data, total 78 attribute data, morphological features with the corresponding stratigraphic information,of craters were extracted for age estimation model.
40 morphological data of craters were calculated with Chang’E data referring to the lunar impact craters database published by LPI[1]; 
38 stratigraphic information of craters were extracted from the 1:5,000,000 Lunar Geologic Renovation (2013 edition) produced by the U.S. Geological Survey. 
What needs illustration is that the value of 38 stratigraphic information refers to the percent of the stratum in the crater.
The details of morphological features and the stratigraphic information can be found in [feature_description.csv](https://github.com/hszhaohs/DeepCraters/blob/master/age_estimation/attribute_headers/feature_description.csv). What needs illustration is that the first 40 features indicate the morphological attribute and the last 38 features indicate thr stratigraphic information. The morphological features refer to the lunar impact craters database published by LPI[1], and some age-independent features (like, Latitude, Longitude, et. al.), repetitive features (like, Diameter [km] and Radius [km], Radius [m]), and some incomplete features are removed. The stratigraphic features refer to the 1:5,000,000 Lunar Geologic Renovation (2013 edition) produced by the U.S. Geological Survey. 

# Experiment detail
We use Adam optimizer with learning rate schedule. The inital learning rate is decayed by the factor of 0.2 at 4/5 of entire epochs. 
The basic parameters are listed below:
* Learning rate = 0.0003
* Epochs = 10
* Batch size = 32
* Weights regularization = 0.0001

**NOTE:** 12 CNN models are optional [Resnet50, Resnet101, Resnet152, Senet, se_Resnet50, se_Resnet101, se_Resnet152, se_Resnext101, Polynet, Inceptionv3, DPN68b and Densenet201].

# To Run
## Training
To train age estimation model with Resnet50.

`python train_moon_age_estimation.py -a=resnet50 -m=mt -b=32 --gpu=0 --lr=0.0003 --epochs=10`

## Testing
To test age estimation model with Resnet50.

`python test_moon_age_estimation.py -a=resnet50 -m=mt -b=32 --gpu=0`

## Predicting
To predict the age of detected craters using the trained age estimation model with Resnet50.

`python pred_dete_moon_age_estimation.py -a=resnet50 -m=mt -b=32 --gpu=0`


# References
[1] Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." Advances in neural information processing systems. 2017.  
[2] <https://github.com/CuriousAI/mean-teacher>
