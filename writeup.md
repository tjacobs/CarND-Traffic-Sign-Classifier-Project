#**Traffic Sign Recognition** 

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test/test01.jpg "Traffic Sign 1"
[image5]: ./test/test02.jpg "Traffic Sign 2"
[image6]: ./test/test03.jpg "Traffic Sign 3"
[image7]: ./test/test04.png "Traffic Sign 4"
[image8]: ./test/test05.png "Traffic Sign 5"
[image9]: ./test/test06.png "Traffic Sign 6"
[image10]: ./test/test07.png "Traffic Sign 7"
[image11]: ./test/test08.png "Traffic Sign 8"
[image12]: ./test/test09.png "Traffic Sign 9"
[image13]: ./test/test10.png "Traffic Sign 10"

### Rubric Points
| Rubric Point            	| Work Done
|:---------------------:|:---------------------------------------------:| 
|Submission Files| I've checked that I've included all files. 
|Data Set Summary| Data 
|Exploratory Visualization| I've chosen 20 random images from the training set and displayed them along with their text label descriptions.
|Preprocessing| I've resized the images.
|Model Architecture| LeNet
|Model Training| 10 Epochs.
|Solution Approach| Added dropout.
|Acquiring New Images| I used Google Maps street view to move around Berlin and take screenshots of traffic signs in Germany.
|Performance on New Images| 
|Model Certainty, Softmax Probabilities|
|Augment Training Data|
|Analyze New Image Performance| I measured precision and recall.
|Visualize Softmax Probabilities|
|Visualize Neural Network Layers|

---

###Data Set Summary & Exploration

####1. Summary.

The code for this step is contained in the second code cell of the notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27839 images
* The size of test set is 12630 images
* The shape of a traffic sign image is 32 by 32 pixels
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

The code for this step is contained in the third code cell of the  notebook.  

First I just pick 48 images at random out of the training set and display them, along with their corresponding label. This helps me get a look into what the signs generally look like.

Then, I plot a histogram of the frequency of each class, i.e. how many images in the training set there are for each label. You can see this here:

# TODO
![Histogram][image1]

As you can see, it's not evenly distributed. We may need to jitter some of the less frequently occuring classes to balance it out for the model's predictions.

###Design and Test a Model Architecture

####1. Preprocessing images

The code for this step is contained in the fourth code cell of the  notebook.

**Shuffle.** When initially investigating the data, I noticed that when plotting the images in their natural order, the same image was basically repeated over and over - because the images come from frames in videos, so they're similar images from frame to frame. So the first step is to shuffle the data before we split it to validation sets, so that it's balanced.

**Greyscale.** The second step is to convert the images to greyscale, because it was recommended to do so to improve performance in the LeNet paper, and I confirmed this with a validation accuracy boost after greyscaling. Here is an example of a traffic sign image before and after grayscaling.

![Greyscale][image2]

**Normalize.** As a last step, I normalized the image data because when looking at the sign photos, there are many that look like they were taken at night, so they are considerably darker than the rest. So we want them to be of similar brightness so that the model can generalise across the sign content regardless of brightness. An example before and after normalizing:

![Greyscale][image2]
![Greyscale][image2]

####2. Data split

The code for splitting the data into training and validation sets is contained in the fifth code cell of the notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using sklearn's _train\_test\_split_ function. 

My final training set had 11,401 images. My validation set and test set had 2,851 and 12,630 images.


### TODO
The sixth code cell of the notebook contains the code for augmenting the data set. I decided to generate additional data because the class frequency was so skewed. To add more data to the the data set, I randomly scaled and shifted images from the less frequent classes to create new images for those classes.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is:
### TODO

####3. The Model

The code for my final model is located in the seventh cell of the notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Layer 1: Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Layer 2: Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten					| Outputs 400								|
| Layer 3: Fully connected		| Outputs 120								|
| ReLU		| 								|
| Layer 4: Fully connected		| Outputs 84								|
| ReLU		| 								|
| Layer 5: Fully connected		| Outputs 43 (n_classes)								|
 
####4. Training

The code for training the model is located in the eighth cell of the notebook. 

To train the model, I used an Adam optimizer. 


# TODO

####5. Solution

--Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.--

The code for calculating the accuracy of the model is located in the tenth code cell of the notebook.

My final model results were:

* Training set accuracy of %
* Validation set accuracy of 92% 
* Test set accuracy of %

The first architecture tried was LeNet.

* What were some problems with the initial architecture?
* 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:

* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test Model on New Images

####1. New German traffic signs

Here are the German traffic signs that I found on the web:

![1][image4]
![2][image5]
![3][image6] 
![4][image7]
![5][image8]
![6][image9]
![7][image10]
![8][image11] 
![9][image12]
![10][image13]

All ten should be pretty easy to classify, as they are all clear.

####2. Predictions

--Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).--

The code for making predictions on my final model is located in the thirteenth code cell of the notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 70km/h      		| 									| 
| Speed Limit 100km/h |  										|
| ?					| 											|
| ?	      		| 					 				|
| ?	      		| 					 				|
| ?	      		| 					 				|
| Road Work			|       							|
| Keep Right ?			|       							|
| Speed Limit 30km/h			|       							|
| Pedestrians			|       							|

### TODO: Find out what those signs are

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of ....

####3. Model Confidence 

--Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making confidence predictions on my final model is located in the forteenth code cell of the notebook.

For the first image, the model is relatively sure that this is a 70km/h speed sign (probability of 0.), and the image is indeed that. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 100km/h   									| 
| .20     				| 										|
| .05					| 											|
| .04	      			| 					 				|
| .01				    |       							|


