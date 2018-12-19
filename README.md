# bag_of_visual_words
Image classification using tiny images and bag of visual words using SIFT

In this project, I have done image classification using two approaches, first is a baseline approach of Tiny Image representation in which each image is resized to 16x16 and entire image is used as feature, this is bad model as it discards hig frequency changes and in not invariant to shifts. 

In second approach, I have computed histogram using bag of SIF features, in which image descriptors are stacked up and clustured using KMeans then histogram in greneated, which is then used as feature for our training model. For training, initially KNN was used, then SVM to inprove aacuracy. 

**File structure :**

**data_load.py** : To import all needed packages and data load, to load the the data use the link below, it contains images both training and testing images of 15 classes. 

https://drive.google.com/file/d/1Pa_bg_CLmKPXBfdU5hu4Jfr8SgKhBnb-/view?usp=sharing

**tiny_image_with_KNN.py** : In this, tiny image represenation is done and layer model is trained and tested using KNN.

**bag_of_SIFT_features_KNN_SVM_training.py** : In this, bag of SIFT features are computed followed by training with KNN and SVM model.

**confusion_matrix.py** : Contains code to create and plot convolution matrix, for result analysis. (src - http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
)



**Analysis** 
 
 **Tiny Image representation with KNN :**
 -- **Time consumed and prediction accuracy** <br>
     Time consumed in training and testing = 1.5 min <br> 
     Accuracy = 18.13% <br><br>
 
 -- **Algorithm descriptions** <br>
     This is a straight forward algorithm, in which we are resizing image to 16X16 and then using entire image as a feature. As stated in problem statement this is just a baseline model, as with resizing to such small dimensions, it discards all of the high frequency image content and is not especially invariant to spatial or brightness shifts. Later we are training model using KNN classifier.<br><br>
     
 -- **Confusion matrix observations**<br>
    Data in confusion matrix, is very sparse, every class is mis labled to other class with a huge percentage, like "store" is mapped to "kitchen" once in every three attempts. There are very few correct classification like "suburb" is correctly labled nearly half the time. <br><br>
 
 **Bag of SIFT features + KNN :** <br>
 
 -- **Time consumed and prediction accuracy**<br>
     Time consumed in KMeans clustering with step size 15 = 17.5 min<br>
     Time consumed in KNN training and testing = 3.5 min<br>
     Accuracy = 51.20%<br><br>
     
 -- **Algorithm descriptions**<br>
    For both second and third combination we have used same set of features for training images. Process of creating these feature involves creating bag of visual words. In this we create a vocabulary that can best describe the image in terms of extrapolable features. It follows three simple steps<br> 
    -- Determination of Image features of a given label, stacking them <br> 
    -- Construction of visual vocabulary by clustering<br>
    -- Histogram creation by frequency analysis <br>
    Post this pre-processing process, we are training a model with KNN classifier, using histogram we created and train_labels. 
    <br>
    Challenges - Initially while computing descriptors for an image, getting huge amount of decriptors per image, which was taking a lot of time later, to stack all these and in clustering by KMeans. In resolution to this, dense sampling is done by taking a step size, in which we are dividing entire image into sections and taking one descriptor from each section. Also, I had to try out on different step size, as taking very small step size results in large number of descriptors (will will take more time in clustering) and very large step size results in very small number of descriptors (which might not capture the image features correctly).<br><br> 
 
 -- **Confusion matrix observations**<br>
     Unlike previous confusion matrix, we can see that in this, many classes are accurately labelled like "mountains" is labelled with 79% accuracy while "forest" are labelled with 70% accuracy, still however for few classes like "kitchen" and "living room" accuracy is low.<br><br>
 
 **Bag of SIFT features + SVM :** <br>
 
 -- **Time consumed and prediction accuracy**<br>
     Time consumed in SVM training and testing (C = 0.8) = 3.5 min<br>
     Accuracy = 57.8%<br><br>
 -- **Algorithm descriptions**<br>
     For this combinations, we are using the same histogram and train_labels we generated previously but insted of KNN we are using one-vs-all SVMs. For getting the best accuracy out of our model, we are training model with different C values, which defines how strongly regulaized our model is, below the respective Cs and accuracys -<br>
     C = 0.1 Accuracy = 56%<br>
     C = 0.8 Accuracy = 57.8%<br>
     C = 1 Accuracy = 57.07%<br>
     C = 4 Accuracy = 57.40%<br>
     C = 10 Accuracy = 56.73%<br>
     <br><br>
 -- **Confusion matrix observations**<br>
     Confusion matrix is slightly better than what we got in previous combination. We can see correct predictions with high accuracy for not just "forest" and "mountains" but also for "street", "suburb" and "highways". Also "kitchen" and "living room" accuracy is also improved compared to previous combination. <br>
