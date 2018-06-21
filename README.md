# Rapid Object Detection using a Boosted Cascade of CNNs
The purpose of this project is to demonstrate the advantages of combining multiple CNNs to a common cascade structure. In contrast to training a single CNN only, the resulting classifier can be faster and more accurate at once. So far, the provided code has been applied successfully to the problem of face detection. It should be straight forward to adapt it to similar use cases though.

Note, this project is about binary(!) classification / detection only. Furthermore, the cascade gets especially fast for highly-unbalanced class distributions.

## Requirements
- Python 3
  - plus some additional pip packages, which are listed in the `requirements.txt` of this repository. see [Installation](#Installation).
- [TensorFlow](https://www.tensorflow.org/) 1.4
  - for installation details see the [TensorFlow manual](https://www.tensorflow.org/install/install_sources)
- [OpenCV](https://opencv.org/) 3.3.1
- other versions as the ones stated above may or may not work as well

## Installation
- clone this repo
- install missing pip packages: `pip3 install -r requirements.txt`
- copy `config_local_sample.py` to `config_local.py` and adjust its settings to your needs

## Datasets
- You can use (almost) any dataset, as long as you store it according to the following folder structure.
- Place your datasets in a new subfolder of the dir which you just specified in the `config_local.py` as the `project_extension_root`
  - the common subfolder must be called `input`
  - each dataset gets its own folder:
      ```
      <project_extension_root>/
        -> input/
          -> <dataset_key_1>/
          -> [..]
          -> <dataset_key_n>/
      ```
  - The top level of such a dataset folder can contain arbitrary extra data, but the images need to be stored in `images/original`.
  Finally, images must be grouped by their labels.
      ```
      <dataset_key_1>/        
        -> images/
          -> original/
            -> label_key_1/
                -> foreground_1.jpg
                -> [..]
                -> foreground_m.png
            -> [..]
            -> label_key_k/
                -> background_1.jpg
                -> [..]
                -> background_j.png
        -> custom_meta_data/
      ```
  - Label keys are defined in `data/db/label.py`. The default labels used for binary classification are `foreground` and `background`.
  - You may split the data inside of the label folders into further subfolders. This additional structure will be ignored, but all images will be read recursively.
      
## Configuration
A lot of further settings can be configured in `config.py`. Descriptions for each setting are provided inside of that file as well. If you want to change a specific setting, you should copy it to your own `config_local.py` file first.

## Run
There are several python scripts you can run from shell. All of them are located in the root dir and named like `run_<action>.py`.
You may want to start with creating ready-to-use samples of your dataset: `python3 run_sampling.py`. A quick look into each of the `run_<action>.py` should be sufficient to get an overview of the remaining possibilities.

## Pre-Trained Models
Currently, there is only one pair of pre-trained models available: one CNN cascade and the associated individual CNN. Both of them were trained for the purpose of face detection:

- [Download the pre-trained CNN cascade for face detection](https://download.johnson145.com/RapidObjectDetectionUsingCascadedCNNs/cascade.zip)
- [Download the pre-trained single CNN for face detection](https://download.johnson145.com/RapidObjectDetectionUsingCascadedCNNs/single.zip)

Training was done using the [Annotated Facial Landmarks in the Wild (AFLW)](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) dataset along with some categories of the [ImageNet](http://www.image-net.org/) dataset. As the associated images are partially-restricted to non-commercial research purposes only, the same restrictions may apply to using the provided pre-trained model files.

In order to use them, you need to extract the downloaded files into the `output_graph_dir`, which is specified in your `config.py`. If you encounter troubles, verify that the `default_evaluation_model_[cascade|single]` setting has not been changed.

## Related Work
This project has been motivated by the following work of Viola and Jones: [Rapid object detection using a boosted cascade of simple features](https://ieeexplore.ieee.org/document/990517/).

Li et al. proposed a similar approach in: [A convolutional neural network cascade for face detection](https://ieeexplore.ieee.org/document/7299170/). However, this project is _not_ a re-implementation of the cascade described by Li et al. If that's what you're actually looking for, you may have a look at [layumi's repository](https://github.com/layumi/2015_Face_Detection) as well as the one of [mks0601](https://github.com/mks0601/A-Convolutional-Neural-Network-Cascade-for-Face-Detection).