# traffic-sign-recognition

This repository contains a PyTorch implementation of a traffic sign detection system using Faster R-CNN. This code can train on the COCO-formatted object detection dataset and evaluate the model's performance.

## Dependencies

- PyTorch
- torchvision
- OpenCV (cv2)
- pycocotools
- albumentations
- numpy

## Steps in Object Detection
1. **Extracting Frames from YouTube Videos with FFMpeg**: Before training the model, videos can be sourced from YouTube, and frames can be extracted using FFMpeg.
2. **Labeling Frames with LabelImg**: For annotating these frames, you'd use LabelImg. This ensures the dataset's richness and accuracy. 
3. **Data Augmentation with Albumentations**: Uses the Albumentations library for on-the-fly data augmentation, enhancing the model's ability to generalize across various traffic scenarios.
4. **Faster R-CNN Implementation**: Utilizes torchvision's Faster R-CNN with a choice between ResNet50 and MobileNet backbones for the detection task.

## Dataset

- **TrafficSignDataset**: A custom dataset class that loads COCO-formatted datasets. It includes utilities to get image annotations, visualize them, and split the dataset into training and test sets.

## Training and Evaluation

- The training loop uses Adam optimizer with a learning rate of 0.001.
- The model can be saved to disk after training and can be loaded for evaluation.
- Provides utilities for computing and printing the mean Average Precision (mAP) of the model on the test set.

## Usage

1. Place your COCO-formatted dataset in the same directory as the script.
2. Update the `root` and `annotation` paths in the `TrafficSignDataset` instantiation in the `main()` function accordingly.
3. By default, the code is set to train mode (`Train=True`). Set it to `False` if you want to load a pre-trained model and evaluate it.
