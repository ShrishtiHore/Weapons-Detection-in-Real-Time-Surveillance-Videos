# Weapons Detection in Real Time Surveillance Videos
This project aims to minimize the police response time by detecting weapons through a live CCTV camera feed. So it alerts the police as soon as it detects any sort of weapons. In our project we are focusing on guns primarily. 

### Code and Resources Used
**Language:** Python 3.8

**Libraries:** pandas, numpy, csv, re, cv2, os, glob, io, tensorflow, PIL, shutil, urllib, tarfile, 
files(google colab), Ordereddict (collections), ElementTree (xml)

**Dataset:** Pistol Dataset by [University of Granada](https://sci2s.ugr.es/weapons-detection)

**Step 1: Gathering Images and Labels**

1. Download the images from the above link. At least 50 images for each class the more the no. of classes the more images.
2. Images with random objects in the backgorund.
3. Various background conditions such as dark, light, indoor, oudoor, etc.
4. Save all the images in a folder called images and all images should be in .jpg format.

**Step 2: Labelling Images**
1. Using LabelImg drag and annotated the guns in the images.
2. The labels will be in PascalVOC format. Each image will have one .xml file that has its labels. If there is more than one class or one label in an image, that .xml file will include them all.
3. And the labels/ annotations will be done.

**Step 3: Setting the Data Systematically**
1. Mount the google drive to google colab.
2. The Directory should be as follows:

object_detection
             └── data
                   ├── images
                   │      ├── image_1.jpg
                   │      ├── image_2.jpg
                   │      └── ...
                   │
                   └── annotations
                          ├── image_1.xml
                          ├── image_2.xml
                          └── ...
                        
3. Split the images into train and test labels (i.e. only the xml files)

object_detection
             └── data
                   ├── images
                   │      ├── image_1.jpg
                   │      └── ...
                   │
                   ├── annotations
                   │      ├── image_1.xml
                   │      └── ...
                   │
                   ├── train_labels //contains the labels only
                   │      ├── image_1.xml
                   │      └── ...
                   │
                   └── test_labels //contains the labels only 
                          ├── image_50.xml
                          └── ...
                          
**Step 4: Import and Install the Required Packages**
1. Install PIL and Cython as they are not pre installed in Google colab.
2. Just import all the packages that were listed above.
3. Tensorflow version should be 1.15.0. for our project.

**Step 5: Preprocessing the Images and Labels**
1. Create 2 csv files for .xml files in each train_labels and test labels folder.
2. Create pbtxt file that will contian the label map for each class
3. The working directory should be as follows:

object_detection/
             └── data/
                   ├── images/
                   │      └── ...
                   ├── annotations/
                   │      └── ...
                   ├── train_labels/
                   │      └── ...
                   ├── test_labels/
                   │    └── ...
                   │
                   ├── label_map.pbtxt
                   │
                   ├── test_labels.csv
                   │
                   └── train_labels.csv
                   
**Step 6: Downloading the Tensorflow model**
1. Tensorflow model contains the object deteection API. So just get from the [official repository](https://github.com/tensorflow/models) by cloning it.
2. Compile Proto buffers and also PATH var should have the directories models/research/ and models/research/slim added.
3. Run a quick test to confirm that the model builder is working properly

**Step 7: Generating TFRecords**
1. The CSVs file names is matched:train_labels.csv and test_labels.csv
2. Current directory is object_detection/models/research
3. Add your custom object text in the function class_text_to_int below by changing the row_label variable (This is the text that will appear on the detected object). Add more labels if you have more than one object.
4. Check if the path to data/ directory is the same asdata_base_url below.

**Step 8: Selecting and downloading a Pre-Trained Model**
1. A pre-trained model simply means that it has been trained on another dataset. That model has seen thousands or millions of images and objects.
2. COCO is a dataset of Common Objects in Context dataset. 
3. Choose a a model that has a low ms inference speed with a relatively high mAP on COCO. The on we are using is ssd_mobilenet_v2_coco. Check the other models from here. You could use any pre-trained model you prefer, but I would suggest experimenting with SSD ‘Single Shot Detector’ models first as they perform faster than any type of RCNN on a real-time video.
4. Download th pretrained model. While training, the model will get autosaved every 600 seconds by default. The logs and graphs, such as, the mAP, loss and AR, will also get saved constantly.
5. The working directory at this point:

object_detection/
           ├── data/
           │    ├── images/
           │    │      └── ...
           │    ├── annotations/
           │    │      └── ...
           │    ├── train_labels/
           │    │      └── ...
           │    ├── test_labels/
           │    │      └── ...
           │    ├── label_map.pbtxt
           │    ├── test_labels.csv
           │    ├── train_labels.csv
           │    ├── test_labels.records
           │    └── train_labels.records
           │
           └── models/           
                ├── research/
                │      ├── training/
                │      │      └── ...
                │      ├── pretrained_model/
                │      ├── frozen_inference_graph.pb
                │      └── ...
                └── ...
                
**Step 9: Configuring the Training Pipeline**
1. ssd_mobilenet_v2_coco.config is the config file for the pretrained model we are using.
2. View the content of the sample config file by running
3. Copy the content of the config file
4. Edit 
- model {} > ssd {}: change num_classes to the number of classes you have.
- train_config {}: change fine_tune_checkpoint to the checkpoint file path.
- train_input_reader {}: set the path to the train_labels.record and the label map pbtxt file.
- eval_input_reader {}: set the path to the test_labels.record and the label map pbtxt file.
- n model {} > ssd {} > box_predictor {}: set use_dropout to true This will be helpful to counter overfitting.
- In eval_config : {} set the number of testing images you have in num_examples and remove max_eval to evaluate indefinitely
5. Final full working directory:

object_detection/
      ├── data/
      │    ├── images/
      │    │      └── ...
      │    ├── annotations/
      │    │      └── ...
      │    ├── train_labels/
      │    │      └── ...
      │    ├── test_labels/
      │    │      └── ...
      │    ├── label_map.pbtxt
      │    ├── test_labels.csv
      │    ├── train_labels.csv
      │    ├── test_labels.records
      │    └── train_labels.records
      │
      └── models/           
           ├─ research/
           │    ├── fine_tuned_model/
           │    │      ├── frozen_inference_graph.pb
           │    │      └── ...
           │    │         
           │    ├── pretrained_model/
           │    │      ├── frozen_inference_graph.pb
           │    │      └── ...
           │    │         
           │    ├── object_detection/
           │    │      ├── utils/
           │    │      ├── samples/
           │    │      │     ├── configs/             
           │    │      │     │     ├── ssd_mobilenet_v2_coco.config
           │    │      │     │     ├── rfcn_resnet101_pets.config
           │    │      │     │     └── ... 
           │    │      │     └── ...                                
           │    │      ├── export_inference_graph.py
           │    │      ├── model_main.py
           │    │      └── ...
           │    │         
           │    ├── training/
           │    │      ├── events.out.tfevents.xxxxx
           │    │      └── ...               
           │    └── ...
           └── ...
           
**Step 10: Tensorboard (optional)**
1. Tensorboard is the place where we can visualize everything that’s happening during training. You can monitor the loss, mAP & AR.
2. We use ngork to get tensorboard on Colab.

**Step 11: Training the Model**
1. model_main.py which runs the training process
2. pipeline_config_path=Path/to/config/file/model.config
3. model_dir= Path/to/training/
4. If the kernel dies, the training will resume from the last checkpoint. Unless you didn’t save the training/ directory somewhere, ex: GDrive.
5. If you are changing the below paths, make sure there is no space between the equal sign = and the path.

**Step 12: Export the Trained Model**
1. the model will save a checkpoint every 600 seconds while training up to 5 checkpoints. Then, as new files are created, older files are deleted.
2. Then by executing export_inference_graph.py to convert the model to a frozen model frozen_inference_graph.pb that we can use for inference. 
3. This frozen model can’t be used to resume training. However, saved_model.pb gets exported as well which can be used to resume training as it has all the weights.
![train](https://github.com/ShrishtiHore/Weapons-Detection-in-Real-Time-Surveillance-VIdeos-/blob/master/train_acc.PNG)

**Step 13: Webcam Inference**
1. To use your webcam in your local machine to inference the model use tensorflow and cv2.
2. You can run the following from a jupyter notebook or by creating a .py file. However, change PATH_TO_FROZEN_GRAPH , PATH_TO_LABEL_MAP and NUM_CLASSES.

**SSD MobileNet V2 (Single Shot MultiBox Detector)**
- This model is a single-stage object detection model that goes straight from image pixels to bounding box coordinates and class probabilities. The model architecture is based on inverted residual structure where the input and output of the residual block are thin bottleneck layers as opposed to traditional residual models. Moreover, nonlinearities are removed from intermediate layers and lightweight depthwise convolution is used. This model is part of the Tensorflow object detection API.
- SSD is a popular algorithm in object detection. It’s generally faster than Faster RCNN. In this post, I will give you a brief about what is object detection, what is tenforflow API, what is the idea behind neural networks and specifically how SSD architecture works.
- The SSD architecture is a single convolution network that learns to predict bounding box locations and classify these locations in one pass. Hence, SSD can be trained end-to-end. The SSD network consists of base architecture (MobileNet in this case) followed by several convolution layers:
![ssd](https://github.com/ShrishtiHore/Weapons-Detection-in-Real-Time-Surveillance-VIdeos-/blob/master/ssd.png)
- By using SSD, we only need to take one single shot to detect multiple objects within the image, while regional proposal network (RPN) based approaches such as R-CNN series that need two shots, one for generating region proposals, one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches.

**Results**

![result1](https://github.com/ShrishtiHore/Weapons-Detection-in-Real-Time-Surveillance-VIdeos-/blob/master/results.gif)
![result2](https://github.com/ShrishtiHore/Weapons-Detection-in-Real-Time-Surveillance-VIdeos-/blob/master/result1.png)
![result3](https://github.com/ShrishtiHore/Weapons-Detection-in-Real-Time-Surveillance-VIdeos-/blob/master/result2.png)
![result4](https://github.com/ShrishtiHore/Weapons-Detection-in-Real-Time-Surveillance-VIdeos-/blob/master/result3.png)
![result5](https://github.com/ShrishtiHore/Weapons-Detection-in-Real-Time-Surveillance-VIdeos-/blob/master/result4.png)

**References**
1. https://stats.stackexchange.com/questions/205150/how-do-bottleneck-architectures-work-in-neural-networks
2. https://medium.com/@techmayank2000/object-detection-using-ssd-mobilenetv2-using-tensorflow-api-can-detect-any-single-class-from-31a31bbd0691
3. https://resources.wolframcloud.com/NeuralNetRepository/resources/SSD-MobileNet-V2-Trained-on-MS-COCO-Data#:~:text=SSD%2DMobileNet%20V2%20Trained%20on%20MS%2DCOCO%20Data&text=Released%20in%202019%2C%20this%20model,box%20coordinates%20and%20class%20probabilities.&text=This%20model%20is%20part%20of%20the%20Tensorflow%20object%20detection%20API.
4. https://heartbeat.fritz.ai/real-time-object-detection-using-ssd-mobilenet-v2-on-video-streams-3bfc1577399c
5. https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d
