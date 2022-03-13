# Vision Transformer (ViT) usado para Calibração Automática de Parâmetros Intrínsecos de Câmeras

## Table of contents

- [Requirements](#requirements)
- [Dataset generation](#dataset-generation)
- [Training DeepCalib](#training-deepcalib)
- [Prediction](#Prediction)
- [Undistortion](#undistortion)
- [Evaluation](#Evaluation)
- [Citation](#citation)


## Requirements
- Python 3.7
- Keras 2.8
- TensorFlow 2.8
- OpenCV 4.5.5

## Dataset generation
There is a code for the whole data generation pipeline - folder [link](https://github.com/arianyfranca01/calibration_of_camera_intrinsic_parameters/tree/main/dataset). First you have to download of images in RGB of sun360 dataset using Google drive [link](https://drive.google.com/drive/folders/1ooaYwvNuFd-iEEcmOQHpLunJEmo7b4NM). Then, use the code provided to generate your continuous dataset. Please, do not forget to [cite](https://scholar.google.co.kr/scholar?hl=en&as_sdt=0%2C5&as_vis=1&q=recognizing+scene+viewpoint+using+panoramic+place+representation&btnG=#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3ARJsOQOkTaMEJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den) the paper describing sun360 dataset.

## Training DeepCalib
To train choose you network: SingleNet, ResNet-50 or Vision Transformer (ViT). All the training codes are available in this [folder](https://github.com/arianyfranca01/calibration_of_camera_intrinsic_parameters/tree/main/network_training).

## Prediction
All codes for all the networks are in the folder prediction [link](https://github.com/arianyfranca01/calibration_of_camera_intrinsic_parameters/tree/main/prediction). And all weights are available on Google Drive [link](https://drive.google.com/drive/folders/1JmV7p6gFEt9mYPBxPcS2QfTUE-dePbLf?usp=sharing).

#### Undistortion
There is a folder whit MATLAB code to undistort multiple images from .txt file. The format of the .txt file is the following: 1st column contains `path to the image`, 2nd column is `focal length`, 3rd column is `distortion parameter`. Each row corresponds to a single image. With a simple modification you can use it on a single image by giving direct path to it and predicted parameters. However, you need to change only `undist_from_txt.m` file, not the `undistSphIm.m`. Folder: [Undistoriton](https://github.com/arianyfranca01/calibration_of_camera_intrinsic_parameters/tree/main/undistortion).

## Evaluation

The code for the evaluation of undistorted images [link](https://github.com/arianyfranca01/calibration_of_camera_intrinsic_parameters/tree/main/metrics) calculates the Mean Square Error (MSE), Structural Similarity Index Measure (SSIM) and Peak-Signal-to Noise Ratio (PSNR) between two sets of images. In this analysis, three sets of images were compared. They are the set of images that received the undistortion with the focal length and distortion parameter values predicted by the model trained with ViT, the set with undistortion generated from the predictions made by the model known in SingleNet and the reference set, which received the undistortion with the labels.


## Citation
```
@inproceedings{bogdan2018deepcalib,
  title={DeepCalib: a deep learning approach for automatic intrinsic calibration of wide field-of-view cameras},
  author={Bogdan, Oleksandr and Eckstein, Viktor and Rameau, Francois and Bazin, Jean-Charles},
  booktitle={Proceedings of the 15th ACM SIGGRAPH European Conference on Visual Media Production},
  year={2018}
}

@inproceedings{xiao2012recognizing,
  title={Recognizing scene viewpoint using panoramic place representation},
  author={Xiao, Jianxiong and Ehinger, Krista A and Oliva, Aude and Torralba, Antonio},
  booktitle={2012 IEEE Conference on Computer Vision and Pattern Recognition},
  year={2012},
}
```
