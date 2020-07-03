# Impact craters detection  
Impact craters detection task uses the R-FCN[1], so the code is clone from py-R-FCN[2].

##  In order to ensure the code running properly, the requirements are showed in the below.
### Requirements: software

0. **`Important`** Please use the [Microsoft-version Caffe(@commit 1a2be8e)](https://github.com/Microsoft/caffe/tree/1a2be8ecf9ba318d516d79187845e90ac6e73197), this Caffe supports R-FCN layer, and the prototxt in this repository follows the Microsoft-version Caffe's layer name. You need to put the Caffe root folder under py-R-FCN folder, just like what py-faster-rcnn does.

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```
2. Python packages you might not have: `cython`, `opencv-python`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

Any NVIDIA GPU with 8GB or larger memory is OK.


## Installation [clone from py-R-FCN repository]
1. Clone the repository  
  ```Shell
  a) Download the repository from <https://github.com/hszhaohs/DeepCraters>
  b) UnZip 'DeepCraters.zip'
  c) cd /DeepCraters/craters_detection
  d) We'll call the directory 'py-R-FCN' into `RFCN_ROOT`
  ```

2. Clone the Caffe repository
  ```Shell
  cd $RFCN_ROOT
  git clone https://github.com/Microsoft/caffe.git
  ```
  [optional] 
  ```Shell
  cd caffe
  git reset --hard 1a2be8e
  ```
  (I only test on this commit, and I'm not sure whether this Caffe is still compatible with the prototxt in this repository in the future)
  
  If you followed the above instruction, python code will add `$RFCN_ROOT/caffe/python` to `PYTHONPATH` automatically, otherwise you need to add `$CAFFE_ROOT/python` by your own, you could check `$RFCN_ROOT/tools/_init_paths.py` for more details.

3. Build the Cython modules
    ```Shell
    cd $RFCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $RFCN_ROOT/caffe
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
   ```


## Demo
1.  To use demo you need to download the pretrained Moon_R-FCN model, please download the model manually from [BaiDu](https://pan.baidu.com/s/1YHmuXvYL29kEH3GhKVbQww) [code: `b6b7`], and put it under `$RFCN/data`. 

    Make sure it looks like this:
    ```Shell
    $RFCN_ROOT/data/rfcn_models/moon_resnet101_rfcn_iter_100000.caffemodel
    ```

2.  To run the demo
  
    ```Shell
    $RFCN_ROOT/tools/demo_rfcn_Moon.py
    ```
    
  The demo performs detection using a ResNet-101 network trained for detection on recognized impact craters with CE-1 data.


## Preparation for Training & Testing
One should organize whose dataset as the format of PASCAL VOC dataset.
The collected datasets (including image dataset [CE-1 and CE-2 DOM/DEM data] and recognized impact craters information) are prepared before converting to the format of PASCAL VOC dataset.
* 1. Download the CE-1 and CE-2 DOM/DEM data from <http://moon.bao.ac.cn/>
* 2. Image mosaic under the specified projection (the details of projection is described in the file of "moon_projection.prj")
* 3. CE-1 and CE-2 DOM and DOM data fusion.
* 4. Download recognized lunar impact craters regulated by the International Astronomical Union (IAU) from <https://planetarynames.wr.usgs.gov/Page/MOON/target>
* 5. Extract the impact craters locating the study area from step 4.
* 6. Crop the fusion image patch containing the impact craters at the center.
* 7. Convert the impact craters into the format of PASCAL VOC dataset.
* 8. The original PASCAL VOC dataset is replaced by the corresponding data obtained at step 7.

-------------

The detailed information of PASCAL VOC dataset can be described as following. [clone from py-R-FCN repository]

1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	tar xvf VOCtrainval_11-May-2012.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	$VOCdevkit/VOC2012                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Since py-faster-rcnn does not support multiple training datasets, we need to merge VOC 2007 data and VOC 2012 data manually. Just make a new directory named `VOC0712`, put all subfolders except `ImageSets` in `VOC2007` and `VOC2012` into `VOC0712`(you'll merge some folders). I provide a merged-version `ImageSets` folder for you, please put it into `VOCdevkit/VOC0712/`.

5. Then the folder structure should look like this
  ```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	$VOCdevkit/VOC2012                    # image sets, annotations, etc.
  	$VOCdevkit/VOC0712                    # you just created this folder
  	# ... and several other directories ...
  ```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $RFCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit0712
    ```

5.  Please download ImageNet-pre-trained ResNet-50 and ResNet-100 model manually, and put them into `$RFCN_ROOT/data/imagenet_models`
6.  Then everything is done, you could train your own model.


## Usage

### Training
To train your own R-FCN detector, use `experiments/scripts/moon_rfcn_end2end.sh`.

```Shell
cd $RFCN_ROOT
./experiments/scripts/rfcn_end2end.sh [GPU_ID] [NET] [DATASET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ResNet-50, ResNet-101} is the network arch to use
# DATASET in {pascal_voc} is the dataset to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

Trained R-FCN networks are saved under:
```
output/<experiment directory>/<dataset name>/
```

### Testing
To test your own R-FCN detector, use `tools/rfcn_test_Moon_Detect.py`.  
Output is written underneath `$RFCN_ROOT/dete_output_{dataname}`.


## Misc

Tested on Ubuntu 16.04 with a Titan X GPU and Intel(R) Core(TM) CPU i7-5930K @ 3.50GHz

## Reference
[1] Dai, J., Li, Y., He, K., et al. R-FCN: Object detection via region-based fully convolutional networks. In NIPS. (2016).  
[2] <https://github.com/YuwenXiong/py-R-FCN>
