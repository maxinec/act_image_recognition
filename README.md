# ACT Driving Simulator Image Recognition #
This project contains custom datasets and annotations used to generate YOLO (You Only Look Once) weights for detecting driver positions within a vehicle.

![Yolo demo](output.gif)


## General Workflow ##
1. Gather images and resize them to a smaller format.  See the "Adding images to the dataset" section for more details.
2. The annotations are generated with the CVAT tool (https://cvat.org) and then exported in YOLO 1.1 format.  For compatibility with YOLO, only rectangular bounds should be used.
3. Use YOLO to train against the dataset.  The training run will provide a general mosaic picture of results.  You can also run the weights against individual sets of images with a pre-defined confidence rate.  See the Running Yolo section for more information.
4. To improve the model, add more samples to the dataset and re-train.  Note that you can re-use the weights generated by previous runs to improve upon each run.

## Project Structure ##
```
imagesets
imagesets/act_simulator     # Photos from ACT simulator videos
imagesets/dmd_dataset       # Photos from the DMD dataset project (https://dmd.vicomtech.org/)
imagesets/stock             # Photos taken from Google search
yolo/train.yaml.template    # YOLO settings template
yolo/detect_collection      # Folder for detection-only runs
yolo/results/detect         # Folder for results from YOLO's detect script
yolo/results/train          # Folder for results from YOLO's train script
```

## Setup ##
1. Check out https://github.com/ultralytics/yolov5 and follow the requirements setup instructions.
2. Create directories for the yolo dataset.  For this example, I am creating:
```
yolo/train
yolo/train/images
yolo/train/labels
yolo/test
yolo/test/images
yolo/test/labels
```
3. Copy yolo/train.yaml.template to yolo/train.yaml and fill in the full path of your train and test folders
4. Build your training and testing data out of the imagesets directory.  With YOLO annotations, there is one text file for each image file.  As long as the names for the different images/labels are different, you can mix and match from the different datasets.  For example, your training could look like:
```
yolo/train/images/act_dataset_1.png
yolo/train/images/dmd_dataset_1.png
yolo/train/images/1.jpeg
yolo/train/labels/act_dataset_1.txt
yolo/train/labels/dmd_dataset_1.txt
yolo/train/labels/1.txt
```

## Adding images to the dataset ##
* This repo uses 640px width for images.  The resizing is so that YOLO runs faster.
* webp files are not accepted by Yolo
* Image names should be unique in all of the dataset so that we can mix and match images.
* CVAT is a great tool for generating annotations for the image files.  In the YOLO format, there is one text file for each image, so you can create new annotation tasks for each batch of images that you add in and still have the annotations work with the full set.  Note: if you create a CVAT project, the list of labels are shared between different tasks.
* The current classes being detected are: ['steering wheel', 'hand'].  When we add more classes, the yaml file will also have to be updated.


## Running Yolo ##
Run Yolo from the repo checked out in the setup.  The relevant files are: train.py and detect.py.

**To generate your YOLO weights:**
```
python train.py --img 640 --batch 4 --epochs 100 --data <train.yaml file path> --weights <weights file> --project <yolo/results/train directory>
```

There are some choices for your yolo weights file.  In the yolo project, there is a yolov5s.pt that can be used.  Alternatively, you can use the weights generated by previous Yolo runs.  They can be found in yolo/results/train/exp<experiment number>/weights.

For some more explanation about the parameters: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9


**To run your model against specific images or videos:**
```
python detect.py --source <directory or image/video file> --weights <weights file> --conf <confidence level> --project <yolo/results/detect directory>
```
This will generate a copy of your images under yolo/results/detect/exp<experiment number>/ with the detections rectangles.
Note: the default confidence level is 0.25.
