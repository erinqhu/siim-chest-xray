# SIIM-ACR Pneumothorax Classification and Segmentation
This is a featured prediction competition host by Society for Imaging Informatics in Medicine (SIIM) on Kaggle (https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview).
## Dataset
SIIM anonymizes 10,000+ and 3,000+ 1024 by 1024 DICOM-formatted radiographs (~2G) for training and test datasets, respectively. `train-rle.csv` file provides labels in the form of run-length-encoded(RLE) masks for most of the images (there exists image name mismatches).<br><br>
_Mask_ <br>
__-1:__ images without pneumothorax have a mask value of -1<br>
__387620 23 996 33 986 43 977 51 968 58 962 65 956 70 952 74 949 76 946 79:__ pixel, num pixels to the right; pixel, num pixels to the right; pixel, num pixels to the right; ...<br>
i.e.: Since our reading order is first left to right, then up to down. In the above sample mask, pneumothorax infected area starts at pixel _387620_ and ends at the 23rd pixel after it.<br>
## Models
* __Simplified Classification Model:__ simplified the project to a binary classification problem by converting all the -1 masks to _0's_ and others to _1's_. _0_ represents the image is nonpneumothorax, while _1_ means the image has pneumothorax.
* __Full Classification Model (ongoing):__ based on the simplified model, the full model identifies infected pixels for pneumothorax images.
## Code
`DownloadDicomData.ipynb` Downloading zipped images from cloud (DropBox) by replacing the _dl=0_ with _dl=1_ of the last part of the file link.<br>
`split_train_dataset.ipynb` Spliting 20% of training dataset to valid dataset, as test dataset is not labeled.<br>
`DicomToJpeg.ipynb` Converting images from DICOM to Jpeg format for both training and test datasets.<br>
`siim_show_mask.ipynb` Marking infected pixels on training images. <br>
`imageCsvLabelMatch.ipynb` Matching all the labels/masks with corresponding images
`siim_cnn.ipynb` Training models using Keras with modified VGG16.<br>
`siim_fastai.ipynb` Training models using fastai (PyTorch) with Resnet34.
## Results
__Keras with VGG16:__ achieves 79.02% accuracy after ten epochs of training with the best accuracy appears at the 6th epoch.<br>
![result_cnn](https://github.com/erinqhu/siim-chest-xray/blob/master/results/result_cnn.png)
__fastai with Resnet34:__ achieves over 85% accuracy after five epoch of training.
![result_fastai](https://github.com/erinqhu/siim-chest-xray/blob/master/results/result_fastai.png)
<br><br>
_Thoughts:_
* `fastai with Resnet34` model trains with higher resolution (224x224) images, while `Keras with VGG16` model uses 150 by 150 images to speed up training time
* `fastai with Resnet34` model uses `max_lr=slice(start_lr, end_lr)` varying learning rates in different layers, while `Keras with VGG16` model sticks with a single learning rate.
