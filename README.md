# Weapons Detection in Real Time Surveillance Videos
This project aims to minimize the police response time by detecting weapons through a live CCTV camera feed. So it alerts the police as soon as it detects any sort of weapons. In our project we are focusing on guns primarily. 

### Code and Resources Used
**Language:** Python 3.8
**Libraries:** pandas, numpy, csv, re, cv2, os, glob, io, tensorflow, PIL, shutil, urllib, tarfile, files(google colab), Ordereddict (collections), ElementTree (xml)
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





