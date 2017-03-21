#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plot.png "Visualization"
[image2]: ./normalise.png "Normalise"

[image4]: ./test/test01.jpg "Traffic Sign 1"
[image5]: ./test/test02.jpg "Traffic Sign 2"
[image6]: ./test/test03.png "Traffic Sign 3"
[image7]: ./test/test04.png "Traffic Sign 4"
[image8]: ./test/test05.png "Traffic Sign 5"
[image9]: ./test/test06.png "Traffic Sign 6"
[image10]: ./test/test07.png "Traffic Sign 7"
[image11]: ./test/test08.png "Traffic Sign 8"
[image12]: ./test/test09.png "Traffic Sign 9"


### Rubric Points
| Rubric Point            	| Work Done
|:---------------------:|:---------------------------------------------:| 
|Submission Files| I've checked that I've included all files. 
|Data Set Summary| I've calculated totals of examples and labels. 
|Exploratory Visualization| I've chosen random images from the training set and displayed them along with their text label descriptions, and plotted distribution of labels.
|Preprocessing| I've normalised the images using OpenCV.
|Model Architecture| My model is based on LeNet.
|Model Training| I've trained it for 50 Epochs.
|Solution Approach| I adapted LeNet. Normalising the images helped greatly, especially with calculating top_5 softmax probablities for confidence evaluation later on.
|Acquiring New Images| I used Google Maps street view to move around Berlin and take screenshots of traffic signs in Germany, and searched on Google Images.
|Performance on New Images| Calculated.
|Model Certainty, Softmax Probabilities| I've calculated softmax probabilities for predictions.

---

###Data Set Summary & Exploration

####1. Summary.

The code for this step is contained in the second code cell of the notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27,839 images
* The size of test set is 12,630 images
* The shape of a traffic sign image is 32 by 32 pixels
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

The code for this step is contained in the third code cell of the  notebook.  

First I just pick 48 images at random out of the training set and display them, along with their corresponding label. This helps me get a look into what the signs generally look like.

Then, I plot a histogram of the frequency of each class, i.e. how many images in the training set there are for each label. You can see this here:

![Histogram][image1]

###Design and Test a Model Architecture

####1. Preprocessing images

The code for this step is contained in the fourth code cell of the  notebook.

**Shuffle.** When initially investigating the data, I noticed that when plotting the images in their natural order, the same image was basically repeated over and over - because the images come from frames in videos, so they're similar images from frame to frame. So the first step is to shuffle the data before we split it to validation sets, so that it's balanced.

**Normalize.** As a last step, I normalized the image data because when looking at the sign photos, there are many that look like they were taken at night, so they are considerably darker than the rest. So we want them to be of similar brightness so that the model can generalise across the sign content regardless of brightness. An example before and after normalizing:

![N][image2]

####2. Data split

The code for splitting the data into training and validation sets is contained in the fifth code cell of the notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using sklearn's _train\_test\_split_ function. 

My final training set had 11,401 images. My validation set and test set had 2,851 and 12,630 images.


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

The code for training the model is located in the 8th cell of the notebook. 

To train the model, I used the Adam optimizer. Batch size set to 128, and learning rate at 0.001.


####5. Solution

The code for calculating the accuracy of the model is located in the 9th code cell of the notebook.

My final model results were:

* Validation set accuracy of 99% 
* Test set accuracy of 92%

The first architecture tried was based on LeNet. 

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

All should be pretty easy to classify, as they are all clear, and are the type of signs that are present within the training set.

####2. Predictions

The code for making predictions on my final model is located in the 11th code cell of the notebook.

Here are the results of the prediction (bold is incorrectly predicted):

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 70km/h      		| 		 Speed Limit 70km/h | 
| Speed Limit 100km/h |  				**Speed Limit 30km/h** |
| Right-of-way			|       	 Right-of-way |
| Keep Right 			|       	Keep Right |
| Speed Limit 30km/h			|       Speed Limit 30km/h |
| Right-of-way	   			   |       **Traffic signals** |
| Priority road			|       Priority road |
| Priority road			|       Priority road |
| Ahead only           |       Ahead only          |
The model was able to correctly guess 7 of the 9 traffic signs, which gives an accuracy of 78%. Not as good as the test set, but it is a small set.

####3. Model Confidence 

The code for making confidence predictions on my final model is located in the 13th code cell of the notebook.

For the first image, the model is relatively sure that this is a 70km/h speed sign, and the image is indeed that. The top five softmax probabilities were:

| Probability         	|     Prediction	        | Softmax | 
|:---------------------:|:---------------------------------------------:|:----:|
| 77%         		 | Speed Limit 70km/h     | [ 0.76599997  0.228       0.006       0.          0.        ]  |
| 100%    			 	 | 		**Speed Limit 30km/h** | [ 1.  0.  0.  0.  0.] 
| 100%				    |      Right-of-way | [ 1.  0.  0.  0.  0.] 
| 100%					 | 	Keep Right	| [ 1.  0.  0.  0.  0.] 
| 100%	      			 | 	Speed Limit 30km/h| [ 1.  0.  0.  0.  0.] 
| 52%	   			   |       **Traffic signals** | [ 0.52899998  0.322       0.149       0.          0.        ] 
| 100%			|       Priority road | [ 1.  0.  0.  0.  0.] 
| 100%			|       Priority road | [ 1.  0.  0.  0.  0.] 
| 100%           |       Ahead only          | [ 1.  0.  0.  0.  0.] 
