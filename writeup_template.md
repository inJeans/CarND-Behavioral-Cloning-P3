#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* data_augmentation.py containing code to read and augment training data
* drive.py for driving the car in autonomous model
* model.h5 containing a trained convolution neural network 
* writeup_report.md

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Training is executed simply with a
```sh
python model.py
```
That of course assumes you have first activated the `carnd-term1` virtual environment.

By default the training script will look for a `.npy` file which contains the preprocessed images. If the file exists it will be loaded in to memory for training (`NEED TO PUT INFO ABOUT RELEVANT FLAGS HERE`). This approach is much quicker but requires a significant amount of system memory to be able to be executed successfully. If the file is not found we default to using the online image generator which will instead stream the images from disk. I found this training approach to be a bit slower as the CPU was anable to feed my GPU fast enough to optimise efficiency on the device. In any case both options are available and should produce similiar* results.

***NOTE**: The online and preprocessing techniques will produce slightly different results as the online approach randomly applies augmentations at runtime. This means that the training algorithm is far less likely to see the same training batches as it cycles through each epoch. The preprocessing technique, however, processes images once and will use the same data for the entire training cycle.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and a depth of 6 (model.py lines 34, 36) 

The model includes RELU layers to introduce nonlinearity (code line 34, 36), and the data is normalized in the model using a Keras lambda layer (code line 33). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). I used the stadard 80/20 train/val split.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well as driving the track backwards to reduce a bias to one direction. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to produce the simplest model possible that was able to navigate the course. My intent was to instead see if I could improve the model by simply diversifying my dataset.

My first step was to use a convolution neural network model similar to the simple LeNet type architecure we have seen in the course material. I thought this model might be appropriate because it seems to be the simplest network that has been able to produce some useful classifcation results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I employed early stopping in the training. That is, I trained for fewer epochs so the model did not have the chance to overfit.

Then I reduced the edpth of the convolution filters to minimise the learning capacity of the network.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I simply gathered more data around these areas of the track. Making sure to gather data from the cra travelling in both directions around the course, to ensure a balanced dataset.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 169x320x3 RGB image   					    |
| Normalise				| Simple max/min normalisation                  |
| Convolution 5x5     	| 1x1 stride, same padding, 6 filters         	|
| RELU					|												|
| Max pooling	      	| 2x2 stride                    				|
| Convolution 5x5	    | 1x1 stride, same padding, 6 filters    		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6    				|
| Dropout               | dropout rate 0.2                              |
| Fully connected		| outputs 120       							|
| RELU					|												|
| Fully connected       | outputs 84                                    |
| RELU					|												|
| Fully connected   	| output 1    									|
|						|												|


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
