# DeepCraters 
Lunar impact craters identification and age estimation with Chang'E data by deep and transfer learning

DeepCraters is a pipeline for training a convolutional neuralnetwork (CNN) 
to identify impact craters on the Moon and training a dual-channel impact 
crater age estimation model to classify the impact craters identified before.

## Getting Started

### Overview

The DeepCraters pipeline trains a impact craters identified model and 
age estimatiom model using data derived from CE-1 and CE-2 DOM and DEM 
image data and catalogue of craters.  The code is divided into two parts. 
The first trains a impact craters identified model with R-FCN [the details 
can be found in the subfile `craters_detection/`]; The second trains a 
dual-channel impact crater classification model using craters with 
constrained ages and craters detected without age.

### Data Sources

- I. http://moon.bao.ac.cn/ [CE-1 and CE-2 DOM and DEM data can be obtained from this website]  
- II. https://planetarynames.wr.usgs.gov/Page/MOON/target [recognized lunar impact craters regulated by the IAU can be obtained from this website]  
- III. https://www.lpi.usra.edu/lunar/surface/ [lunar craters with constrained ages aggregated by the LPI can be obtained from this website]  
- IV. https://astrogeology.usgs.gov/search/map/Moon/Geology/Lunar_Geologic_GIS_Renovation_March2013 [Lunar 1:5M Geologic Map, 38 stratigraphic information of craters extracting from, can be obtained from this website]

**NOTE:** All the craters with constrained ages in 'II' are contained in 'III'.

The craters used for 'ctaters detection' can be find in `/craters_detection/data_list/`, 
and the craters with constrained ages used for 'age estimatiom' can be find in `/age_estimation/data_list/`.

### Running DeepCraters

Each part of DeepCraters has the corresponding script: 
 - part 1 (craters_detection):  
  `RFCN_ROOT/experiments/scripts/moon_rfcn_end2end.sh` for build and train the detection model,  
  `RFCN_ROOT/tools/rfcn_test_Moon_Detect.py` to generate a crater atlas of study area.  
  The more details can be find in [README for Craters Detection](https://github.com/hszhaohs/DeepCraters/blob/master/craters_detection/README.md).
  
 - part 2 (age_estimation):  
  `train_moon_age_estimation.py` for build and train the age estimation model,  
  `pred_dete_moon_age_estimation.py` to predict the age of all craters detected in part 1.  
  The more details can be find in [README for Age Estimation](https://github.com/hszhaohs/DeepCraters/blob/master/age_estimation/README.md).

We recommend copying these scripts into a new working directory (and appending
this repo to your Python path) instead of modifying them in the repo.

## License

DeepCraters is free software made available under the MIT License. For details see
the LICENSE.md file.
