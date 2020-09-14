# **Behavioral Cloning** 

## Writeup Template

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
[image8]: ./images/NVIDIA.jpg "Nvidia"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia model from the paper "End to End Learning for Self-Driving Cars" as my skeleton model.
![alt text][image8]

My model consists of 5 convolution layers with a combination of 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 70-98). Then 4 more fully connected layers are followed ranging from output sizes (100-50-10-1). More details will be given in the final architecture. 

The model includes RELU layers to introduce nonlinearity after each convolutional layer (code line 70-83), and the data is normalized in the model using a Keras lambda layer (code line 65). 


#### 2. Attempts to reduce overfitting in the model

To reduce the effect of overfitting, I decreased the epochs from 10 to 5. At the beginning, the validation loss kept oscillating decreasing and then increasing which were clear signs of overfitting so I kept decreasing the epochs until I ensured the validation loss decreases at each and every single epoch.

The model contains dropout layers in order to reduce overfitting (model.py lines 84). 

 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

No of epochs= 5
Optimizer Used- Adam
Learning Rate- Default 0.001
Validation Data split- 0.2
Generator batch size= 32
Loss Function Used- MSE(Mean Squared Error).

#### 4. Appropriate training data

I used the Udacity dataset provided as training data. For more details, see final section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Here I will adress the steps and The overall strategy for deriving the final model architecture.

My first step was just to train a single simple one layer regression network to make sure the system is working and to test on the simulator. It worked with poor results. 

Then I used the LeNet framework (See modelLeNet.py), it produced good normal driving behaviour but the vehicle occasional drives outside the road.

Finally, I used the NVidia network and it produced solid results and normal driving behavoiour. I just had to do some tweeks on the network so it would produces even better results.

At the beginning of the NVidia network I normalized the data using a Keras lambda layer (code line 65).
Then I trimmed the image to see only the section of the road cropping clutter and undesired data (e.g. hood of the car, trees and any background scenery) (model.py line 67).
I then used the exact NVIDIA layers till the end and finally added just one fully connected layer to output the final steering angle prediction. Here are the details:

The first convolutional layer with filter depth as 24 and filter size as (5,5) with (2,2) stride followed by RELU activation function
Moving on to the second convolutional layer with filter depth as 36 and filter size as (5,5) with (2,2) stride followed by RELU activation function
The third convolutional layer with filter depth as 48 and filter size as (5,5) with (2,2) stride followed by RELU activation function
Next we define two convolutional layer with filter depth as 64 and filter size as (3,3) and (1,1) stride followed by ELU activation funciton
Next step is to flatten the output from 2D to side by side
Here we apply first fully connected layer with 100 outputs
Here is the first time when we introduce Dropout with Dropout rate as 0.25 to combact overfitting (later removed, did not provide significant improvement)
Next we introduce second fully connected layer with 50 outputs
Then comes a third connected layer with 10 outputs
And finally a single output layer that predicts the final steering angle

The final step was to run the simulator to see how well the car was driving around track one. 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road (see video.mp4).

#### 2. Final Model Architecture

The final model architecture (model.py lines 62-98) consisted mainly of a nomrmalization layer followed by 5 convolution layers followed by 4 fully connected layers:

Here is the architecture:

```python
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# # trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))  
# Add 5 convolutional layers followed by 3 fully connected layers

#layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Conv2D(24,5,5, subsample=(2,2),activation="relu"))

#layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Conv2D(36,5,5, subsample=(2,2),activation="relu"))

#layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Conv2D(48,5,5, subsample=(2,2),activation="relu"))

#layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64,3,4, activation="relu"))

#layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64,3,4, activation="relu"))
#model.add(Dropout(0.25))
#flatten image from 2D to side by side
model.add(Flatten())

#layer 6- fully connected layer 1
model.add(Dense(100))

#layer 7- fully connected layer 1
model.add(Dense(50))

#layer 8- fully connected layer 1
model.add(Dense(10))

#layer 9- fully connected layer 1
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

I utilized the provided Udacity data as training data. It contains 9 laps of track 1 with recovery data. It was pretty decenet and gave strong results so I did not need further training.

I decided to split the dataset into training and validation set using sklearn preprocessing library. I decided to keep 20% of the data in Validation Set and remaining in Training Set

I used a generator to generate the data to avoid loading all the images in the memory and instead generate it at the run time in batches of 32.

I randomly shuffled the data set and put 20% of the data into a validation set. 


 I preprocessed this using a lambda layer for normalization and did some cropping to focus only on pixels of the road that will be useful for immitating driving behaviour.
 
 ```python
from keras.layers import Lambda, Cropping2D
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# # trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))  
```


 The validation set helped determine if the model was over or under fitting. I used 5 epochs and made sure there wasn't any osicllation in terms of the validation loss and that it was always decreasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.
