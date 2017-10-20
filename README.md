# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dist_train]: ./writeup_files/dist_train.png "dist_train"
[unrotated]: ./writeup_files/unrotated.png "unrotated"
[rotated]: ./writeup_files/rotated.png "rotated"

[german1]: ./download_test_data/01000.png "german1"
[german2]: ./download_test_data/02000.png "german2"
[german3]: ./download_test_data/03000.png "german3"
[german4]: ./download_test_data/04000.png "german4"
[german5]: ./download_test_data/05000.png "german5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

I am submitting a zip file that contains the code, writeup, and all the supplemtary files.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python commands to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

The training set consists of 43 kinds of traffic signs. The occurrances of each signs in the training sample are shown in the following plot.

![alt text][dist_train]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, the training data was augmented with rotated images of the original. The purpose of this operation was to reduce the overfitting and increase the amount of the training data. Each image is rotated by 4 degrees and 8 degrees, clockwise and counterclockwise. The rotation was performed by the functions from Image module in Pillow package.

Here is an example of a traffic sign image before and after a rotation operation.
![alt text][unrotated]
![alt text][rotated]

After that, I normalized the image data because the neural network tends to learn faster with the normalized inputs. It also reduces the chance of getting stuck in local minima. Each pixel x was replaced by (x - 128)/128 for normalization. 

If I was given more time, I would have tried converting the image to grayscale and see if it leads to a more efficient learning. Also I would have tried scaling the images to zoom in and out to achieve some scale invariance in the model. I would have tried changing aspect ratios, which might have had the effect of seeing the signs from different angles.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32    				|
| Fully connected		| 400 nodes, relu, 50% dropout                  |
| Fully connected		| 200 nodes, relu, 50% dropout                  |
| Softmax				| 43 nodes                                      |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The loss function used for the training is cross-entropy calculated from the softmax output and the one hot data, i.e.,

$$
\textit{D}(\hat{\textbf{y}}, \textbf{y}) = -\sum_{j} \textbf{y}_j \ln \hat{\textbf{y}_{j}},
$$

where $$\textbf{y}$$ is a vector of one hot data and $$\hat{\textbf{y}}$$ is the softmax output.

The function was optimized with the by [Adam](https://arxiv.org/pdf/1412.6980v8.pdf), a stochastic optimization algorithm.

Tha batch size was 128, number of epochs 5, and the learning rate 0.001.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.968
* test set accuracy of 0.958

I began the project with the architecture 'LeNet', one defined in the precious lessons. LeNet was successfully used for the recognition of the hand written digits (MNIST). The traffic signs are more complicated than the digits, but they are also two dimensional images, so a similar approach would make sense for a starting point.

The input depth was changed from 1 to 3 to accomodate the RGB color space. The depths of the first convolution layer was increased from 6 to 16 because the traffic sign data seemed more complicated than the MNIST examples. The depth of the second convolution layer was also increased from 16 to 32. Due to these changes, the input to the first fully connected layer was increased from 400 to 800. The sizes of the following hidden layers was increased from 120 to 400, and 84 to 200, and the output size was 43.

I have tried dropout layers on two convolution layers and fully connected layers. The dropout layers on the convolution layers didn't seem to improve the outcome, so I only kept them on the fully connect layers. After trying a few different keep probabilities, I have chosen the value to be 0.5. The learning rate of 0.001 was chosen because lower values led slow leaning, and higher values seemed to underfit.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][german1]
![alt text][german2]
![alt text][german3]
![alt text][german4]
![alt text][german5]

The images have different sizes than the training images. They are all converted to the size of 32x32. The sign in the second image is not aligned at the center, but the convolution process should be able to deal with the situation. The sign in the fourth image is very dark because of the bright background. I was worried that the network may be confused with this image, but the network identified the sign correctly. There may have been examples of similarly dark signs in the training sample.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry              | No entry                                      | 
| Turn left ahead       | Turn left ahead                               |
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Road work             | Road work                                     |
| Traffic signals       |Traffic signals                                |


The model correctly classified all five examples, with a 100 % accuracy. This seems consistent with the test accuracy of 0.958.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

All five images were correctly categorized with very high probabilities. The first sign is classified as a 'No entry' sign with a great confidence, leaving the possibilities for all other categories to be zero. This is no surprise, as the sign is big and bright, with a high contrast. The second sign is smaller and darker, compared to the first one, and is located off-centered in the frame. This may have added some ambiguity, and led to the non-zero probabilities for other categories, which are still many orders of magnitudes smaller than the first choice of the network. The third and fifth signs had the similar results with the second sign. The forth image is the darkest of all, with the least contrast, and it had a somewhat lower probability of 0.87 for the best match. I think the network might have worked better for the forth image if I had normalized the images for the contrast. The way I normalized was rather simplistic.

![alt text][german1]

|     prob | name                                               |
|---------:|:--------------------------------------------------:|
|         1 | No entry|
|         0 | Speed limit (20km/h)|
|         0 | Speed limit (30km/h)|
|         0 | Speed limit (50km/h)|
|         0 | Speed limit (60km/h)|

![alt text][german2]

|     prob | name |
|---------:|:--------------------------------------------------:|
|         1 | Turn left ahead|
|   5.7e-06 | Keep right|
|   1.2e-07 | Ahead only|
|   2.2e-09 | Go straight or right|
|   2.5e-10 | Turn right ahead|

![alt text][german3]

|     prob | name |
|---------:|:--------------------------------------------------:|
|         1 | Right-of-way at the next intersection|
|   5.2e-15 | Pedestrians|
|   1.7e-15 | Beware of ice/snow|
|   1.4e-21 | Children crossing|
|   1.3e-22 | Dangerous curve to the right|


![alt text][german4]

|     prob | name | 
|---------:|:--------------------------------------------------:|
|      0.87 | Road work|
|      0.13 | Slippery road|
|   0.00028 | Dangerous curve to the right|
|   4.2e-05 | Beware of ice/snow|
|   3.3e-06 | Dangerous curve to the left|

![alt text][german5]

|     prob | name |
|---------:|:--------------------------------------------------:|
|         1 | Traffic signals|
|   3.5e-13 | General caution|
|   1.5e-23 | Road narrows on the right|
|   4.3e-25 | Bumpy road|
|   5.8e-28 | Pedestrians|

