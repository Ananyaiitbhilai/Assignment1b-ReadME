# Fruit Classification

We will be using the fruits-360 dataset to train and test our models.


## Prototyping

Firstly we extract the features of image using Histogram of Oriented Gradients (HOG) for each image in a class. <br>
```hog(fruit_img, orientations=5, pixels_per_cell=(15, 15), cells_per_block=(2, 2), multichannel=True)``` <br>

Given the parameters remain the same, HOG extracts a feature of a fixed length each time. With the above parameters, the length of the feature vector turns out to be 500 


Expand a bit
Orientations tell the number of buckets to be formed during the making of histogram. <br>
Pixels per cell tells the number of pixels to be considered to form a block for making one histogram. <br>
Cells per block tells the number of blocks to be taken in for averaging out the histogram <br>

Once we have the method of feature extraction, we extract feature vectors for all the images in a single class. <br>
Then we take the average of all the feature vectors of one class and we call this average vector as a prototype for the given class. <br>

Therefore for 131 classes, we will be having a total of 131 prototypes. We pickle these prototypes for later use. <br>

Now in ```tester_sklearn.py``` file, we take all the images in the Test folder and test it against our prototypes and make a prediction for the class it belongs to. <br>

We calculate the feature vector of the test image using the same HOG parameters during the training period. Then we find the prototype vector closest to our feature vector of the test image and classify the image accordingly. <br>

### Reason why prototype has a fairly low accuracy

The prototyping is a fairly rudimentary way of making predictions. By averaging out the feature vectors, we lose a lot of information and try to generalize a bit too much. There are images of fruits with image shots from multiple angles(rotated), trying to average out the prototypes of such images would be very wrong. <br>

Hence it performs poorly when an image slightly different from the training set is presented to prototype model.  
  
<img src="https://i.imgur.com/rOAfQuo.png" width="500">  
<img src="https://i.imgur.com/vsVaXZ9.png" width="500"> 
<img src="https://i.imgur.com/IMjbAiz.png" width="500"> 


The overall accuracy of the model was 46.43% when it was tested against the Test dataset.

## K-Nearest Neighbours (KNN)

Firstly we reduce the color channels in our images to one i.e., we make our images grayscale. This is done to reduce the computation needed to develop the model.  

Secondly we extract the features of image using Histogram of Oriented Gradients (HOG) for each image in a class.  
```hog(img, orientations=5, pixels_per_cell=(10, 10),cells_per_block=(1, 1), channel_axis=None)```  

Then we create four arrays called ```x_train, y_train, x_test, y_test```. ```x_train``` contains all the feature vectors of the training set and ```y_train``` contains the corresponding labels for the training set. Similarly we define ```x_test``` and ```y_test```.   

We then use Scikit-Learn KNeighborsClassifier to calculate K-Nearest Neighbours for the test set and make a prediction for each image.  

The way KNN works is that it first K nearest neighbours of a given test image and then counts which class has the highest count. The one with highest count is the prediction made by KNN algorithm.

Output of the KNN running on our test data

![](https://i.imgur.com/c1XbSRr.png)

KNN seems to perform much better than Prototype model...One possible reason could be that in KNN we did not take average of all feature vectors like we did in Prototype Model. We just calculate distance from all the images and select the K shortest distances.

We have implemented our own implementation of the KNN algorithm but it is very slow compared to Scikit-Learn KNeighborsClassifier. Our version of the implementation can be found at ```knn_self.py```

Apart from this, we also have a python script named ```knn_predictor.py``` which takes in a path of image as an argument and returns the prediction as the output.  

Here's how the output looks like

```
Enter the path to the image: /Users/ananyahooda/Desktop/DS250 Assignment 1b/fruits-360_dataset/fruits-360/Test/Tomato Cherry Red/202_100.jpg
Prediction: Tomato Cherry Red
```


### Reason why KNN has more accuracy than prototype and lower than Simple Neural Network
- Prototyping takes the distance of an image with the average euclidian distance calculated from a class rather than all the images in ecery class, which is the case in KNN.
- KNN depends on the neighbuors only. It doesn’t even need a training phase. The target class is the class with more neighbors nearby.
- ANN uses weights and change / transfer the weights of each neuron with respect to the input. So they need a larger training phase to get more accuracy.

## Neural Network

1. Preprocessing of raw image data
2. Feature selection
3. Model development
4. Model validation and error calculation
5. Error prediction

We have divided Our Trainind folder into 2 parts:- Training(80%) and Validation(20%). <br><br>
We use the flattening technique which is used to convert a matrix into a 1-D array.
We resize the image to 32x32.It is critical to resize our images properly because this neural network requires these dimensions. Each neural network will require different dimensions, so just be aware of this. Flattening the data allows us to pass the raw pixel intensities to the input layer neurons easily.  
We get the length of feature vector as 32x32x3. Then we extract the fruit labels and normalize them.<br><br>
We use label binarizer. At learning time, this simply consists in learning one regressor or binary classifier per class. In doing so, one needs to convert multi-class labels to binary labels (belong or does not belong to the class).<br><br>
Our ANN architecture looks like this:
`3072x1024x512`, using *relu* function as activation functions in the hidden layers and *softmax* activation in the output layer.<br><br>

Stochastic Gradient Descent (SGD) optimizer with *categorical_crossentropy* as the loss function. The batch_size controls the size of each group of data to pass through the network.<br> <br>
We run it for 100 epochs and for batch size = 32 on validation dataset and have the the following accuracy and val_accuracy in the 100th epoch.<br>
<img src="https://i.imgur.com/7NCQPij.png" width="500px"> <br>

We then save our model in the file `fruits-360_model_new5.h5` and test this model on our testing dataset.<br>
We get the following precision, recall, F1-score, support:<br>
<img src="https://i.imgur.com/SRkNHKp.png" width="500px">
  
Our Graph of loss v/s no. of Epochs, as number of epochs increases, loss function decreases.Our trainig and validation Loss is depited in the graph<br>
<img src="https://i.imgur.com/wnOoxPt.png" width="500px"><br>
                                                             

We enter a test image `/Users/ananyahooda/Desktop/DS250 Assignment 1b/fruits-360_dataset/fruits-360/Test/Beetroot/129_100.jpg` and get the following output and probability graph:
<img src="https://i.imgur.com/UE9My4x.png">
<img src="https://i.imgur.com/iBgesCz.png" width="500px">
