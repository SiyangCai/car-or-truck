# Car or Truck?

This is a well-known classification problem in the Kaggle deep learning course example. I'm going to use convolutional neural network to build a deep learning model in Python Tensorflow to help classify the sample [car-or-truck dataset](https://www.kaggle.com/datasets/ryanholbrook/car-or-truck) with cross validation.

## Load image Dataset
This dataset has been split into training and validation files. In both files, they are manually classified into cars and trucks. It is very common that people just use the training and validation dataset separately. However, I'm going to merge those files and randomly split them in the training phase. There are actually two ways to load image data:
  * use `imread` in `cv2` package, or
  * use `load_img` and `img_to_array` in `Keras`. 
 
Both of them should give same results. Shuffle the data and double check the shape of image to see if they are correct. Here I use image size as $128 \times 128$ and RGB color mode
