# Temporal Driver Action Localization using Action Classification Methods (TDAL)
This repository includes the implementation of the TDAL framework, a solution for Track 3 Naturalistic Driving Action Recognition of the [NVIDIA AI City 2022 Challenge](https://www.aicitychallenge.org/). The proposed TDAL framework achieves an F1 score of 27.06% in this challenge. <br />

**Important Note:** <br />
For reproducibility, you must use all the code provided in this repo. Using any files from the different version may give different results (e.g., git clone yolov5 from the official repository)  <br />

## Overview 

Temporal driver action localization (TDAL) framework aims to classify driver distraction actions to 18 classes, as well as identifying the start and end time of a given driver action. The TDAL framework consists of three stages: 
**Preprocessing**, , which takes untrimmed video as input and generates multiple clips; **Action Classification**, which classifies the clips; and finally, the classifier output is sent to the **Temporal Action Localization** to generate the start and end times of the distracted actions. <br />
<br />
The proposed framework achieves an **F1 score** of **27.06%** on Track 3, **A2 dataset** of NVIDIA AI City 2022 Challenge. Our paper will be available soon in CVPR workshops 2022. 

## Framework 

![Framework](https://github.com/Shahad24/AICITY2022_Track3_Team95/blob/main/imgs/Image1.png)

## Development Environment 

The code has been tested with the following hardware and software specifications: <br />
  -	Ubuntu 20.04 and 18.04.
  -	Intel Xeon xxxx.
  -	2 GPUs Geforce RTX xxxx with xxxx GB memory. 
  -	Driver version xxxxx.
  -	Docker version xxxxx.  
  -	Cuda xxxx and cudnn xxxx.

## Installation (basic)

This installation step is needed for both [Training](https://github.com/Shahad24/AICITY2022_Track3_Team95/tree/main/slowfast_train) and [Inference](https://github.com/Shahad24/AICITY2022_Track3_Team95/tree/main/slowfast_Inference) steps.

  1. Pull the image and create container for Pytorch 1.8.1, cuda xxxx and cudnn xxxx
  ```bash
  sudo docker run -it --rm --gpus all --shm-size=40g --name pytorch -v 'your home path':/workspace 8feef0e83aed 
  ```   
  2. Install the following dependencies inside the docker container
  ```bash
  apt-get update && apt-get -y install gcc && apt install g++ -y && apt-get update && apt-get upgrade -y && apt-get install -y git
  pip install numpy
  python -m pip install -U matplotlib
  pip3 install -U scikit-learn
  pip install 'git+https://github.com/facebookresearch/fvcore'
  pip install simplejson
  conda install av -c conda-forge -y
  conda install -c iopath iopath -y
  pip install psutil 
  pip install opencv-python 
  pip install tensorboard 
  pip install pytorchvideo
  conda install -c conda-forge moviepy -y
  ```   
---
## Training Action classification model 
The workflow for training action classification model is as follow:
  1. Dataset preparation <br/>
    - **Trimming Videos** the input videos should be a trimmed videos i.e., contains only one action in each video. <br/>
    - **Driver Tracking (pendding)** to detect and track driver spatial location in the video, then crop each video based on the driver bounding box. <br/>
    - **Image Colorization** to increase the size of the dataset, we use one of the synthetic data generation techniques. <br/>
    - **Prepare csv Files** for the training and validation sets.
  2. Download checkpoints from [google drive here]() 
  3. Prepare the configuration file and start training.

**Important Note:** <br />
Since we clean dataset manually to reproduce near same action classification model, you can download the processed data from the google drive link that we sent via email (note this only for authorized people from organizer of AI city challenge). Then you can start from preparing csv files for the training<br />


  ### Driver Tracking (pendding) 
 
<br />

  ### Image Colorization
  To increase the size of the training data set, we perform image colorization [InstColorization](https://github.com/ericsujw/InstColorization) on the entire data from the previous step. We used 'coco_finetuned_mask_256_ffs' checkpoint. Please follow the instruction in [InstColorization](https://github.com/ericsujw/InstColorization) to get the same results.
<br />

  ### Prepare csv Files
  To train [SlowFast](https://github.com/facebookresearch/SlowFast), we need to prepare the dataset to match their format (for more information see [SlowFast kinetics Dataset](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md)) <br />
  - The dataset information should be stored in CSV file without header, must following the below format.
    > path_to_video_1 label_1
  - The CSV files must be moved to data folder in slowfast_train folder. At the end, you should have the following: <br />
    > slowfast_train/data/train.csv ,  slowfast_train/data/val.csv
    
    
  ---
  ### Training
  Since action recognition method is data hungry and we only have few samples per class. We have to stages to get the final action classification model.<br />
  1. In the first stage, we train action classification model using only the Infrared video without the synthetic data.  
  2. In the second stage, we resume training the first stage model but after adding the colored data samples in the train and test csv files.

  **Note:** For reproducing rapidly, we recommend skipping the first stage and start from second stage using the first stage checkpoint “checkpoint_epoch_00440.pyth”. please do not change the checkpoint name.

  <br />

  #### Training setup 
  After installing the basic libraries in **Installation (basic)**. Type the following command 
  ```bash
  cd slowfast_train 
  git clone https://github.com/facebookresearch/detectron2.git 
  pip install cython
  pip install pandas
  pip install -e detectron2
  python setup.py build develop
  apt-get update
  apt install libgl1-mesa-glx -y
  apt-get install libglib2.0-0 -y
  ```   
  
  #### First stage 

  - You need to download “SLOWFAST_8x8_R50.pkl” form [here]()a kinetics 400 dataset pretrained model and move the file to “slowfast_train/checkpoints/SLOWFAST_8x8_R50.pkl”.  <br />
  - Use “slowfast_train/configs/Kinetics/SLOWFAST_8x8_R50_config1.yaml” config file and add the checkpoint path if needed. <br />
  - Run the following command 
  ```bash
  python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R5ـconfig1.yaml DATA.PATH_TO_DATA_DIR “specify the path to the CSV files”
  ```  
  
  #### Second stage 

  - You need either to download “checkpoint_epoch_00440.pyth” pretrained model on the IR video samples or use the last model in the first stage. Then, move the file to “slowfast_train/checkpoints/“checkpoint_epoch_00440.pyth”. <br />
  - Use  “slowfast_train/configs/Kinetics/SLOWFAST_8x8_R50_config2.yaml” config file and add the checkpoint path if needed. <br />
  - Run the following command 
  ```bash
  python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R5ـconfig2.yaml DATA.PATH_TO_DATA_DIR “specify the path to the CSV files”
  ```  
 ---
## Inference 
To use TDAL framework and produce the same result in the leaderboard you need to follow the following steps:
1. Dataset preparation
  - Driver tracking to detect and track driver spatial location in the video then crop each video based on the driver bounding box.
  - Video segmentation to divide the untrimmed video into equal-length clips.
  - Prepare csv file for the feature and classes probabilities extraction.
2. Extracting action clips probabilities.   
3. Temporal localization to get the start and end time for each predicted distracted action in an untrimmed video. 


  ### Driver Tracking
  1. First, you need to configure yolov5 requirements. Using the same docker container please type the following commands:
  ```bash
  cd yolov5
  pip install -U vidgear[core]
  pip install -U scikit-learn
  pip install -U scikit-image
  pip install opencv-python
  pip install -r requirements.txt
  apt-get update && apt-get upgrade -y && apt-get update && apt install libgl1-mesa-glx -y && apt-get install libglib2.0-0 -y
  ```   
  2. Run the following command after specifying the videos path directory
  ```bash
  python yolov5/driver_tracking.py --vid_path 'specify videos path based on the workspace'  --out_file 'specify the path of output videos based on the workspace'
  ```   
  ### Video Segmentation 


  ### Prepare csv file

---
## Acknowledgement
This repository depends heavily on [SlowFast](https://github.com/facebookresearch/SlowFast), [YOLOv5](https://github.com/ultralytics/yolov5), and [InstColorization](https://github.com/ericsujw/InstColorization)

      
## Contact us :
 > If you faced any error please don't hesitate to contact us : <br />
 > Munirahalyahya21@gmail.com <br />
 > Shahadaalghannam@gmail.com <br />
 > Taghreedalhussan@gmail.com <br />
 > loading checkpoints in the google drive may take time since their are big, and may uploaded after our submission
