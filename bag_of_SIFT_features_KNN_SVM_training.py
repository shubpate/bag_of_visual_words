# Classification using Bag of SIFT features + KNN / SVM model

def formatND(l):
        print ("Stacking begins ..")
        vStack = np.array(l[0])
        for remaining in l[1:]:
            vStack = np.vstack((vStack, remaining))
        print ("Stacking done")
        return vStack
    
def create_des_list(X):
    sift = cv2.SIFT()
    des_list = []
    for image in X:
        kp, des = sift.detectAndCompute(image,None)
        des_list.append(des)
    return np.array(des_list)

def create_sampled_des_list(X,step):
    sift = cv2.SIFT()
    des_list =[]
    for img in X:
        cols, rows = img.shape
        kps = []
        for x in range(step, rows, step):
            for y in range(step, cols, step):
                kps.append(cv2.KeyPoint(x,y,step))
        kps, des = sift.compute(img,kps)
        des_list.append(des)
    return np.array(des_list)

def developVocabulary(n_images, descriptor_list):
        mega_histogram = np.array([np.zeros(no_clusters) for i in range(n_images)])  # #n_images x #no_clusters
        old_count = 0
        #print len(descriptor_list)
        for i in range(n_images):
            l = len(descriptor_list[i])
            #print l
            for j in range(l):
                idx = kmeans_ret[old_count+j]
                mega_histogram[i][idx] += 1
            old_count += l
        print "Vocabulary Histogram Generated"
        #print old_count
        return mega_histogram
    

def recognize(model,test_img,step):
    sift = cv2.SIFT()
    #kp, des = sift.detectAndCompute(test_img,None)
    cols, rows = test_img.shape
    kps = []
    for x in range(step, rows, step):
        for y in range(step, cols, step):
            kps.append(cv2.KeyPoint(x,y,step))
    kps, des = sift.compute(test_img,kps)
    vocab = np.array( [[ 0 for i in range(no_clusters)]])
    test_ret = kmeans_obj.predict(des)
    for each in test_ret:
        vocab[0][each] += 1
    lb =  model.predict(vocab)
    return lb

def test_model(model, x_test):
    predictions=[]
    start_time = time.time()
    print ("Testing started ..")
    for image in x_test:
        cl= recognize(model, image, 15)
        predictions.append(cl)
    print ("Testing completed")
    print ("Time taken in testing - {0}".format((time.time() - start_time)))
    return np.array(predictions)


#code for doing clustering over descriptors	
no_clusters = 50
#des_list = create_des_list(train_data)
des_list = create_sampled_des_list(train_data,15)
bov_stack = formatND(des_list)

start_time = time.time()
print ("Clustering Started ...")
kmeans_obj = KMeans(no_clusters)
kmeans_ret = kmeans_obj.fit_predict(bov_stack)
print ("Clustering Completed")
print ("Time taken in clustering - {0}".format((time.time() - start_time)))

# uncomment below lines of code to save the kmeans_obj using pickle
# outfile = open('cluster_data', 'wb')
# pickle.dump(kmeans_ret,outfile)
# outfile.close()

histogram = developVocabulary(1500, des_list)

## Training using KNN model

#KNN
def train_knn(X,Y):
    start_time = time.time()
    print ("Training ...")
    knn_model = KNeighborsClassifier(len(np.unique(Y)))
    knn_model.fit(X,Y)
    print ("Training completed")
    print ("Time taken in training {0}".format((time.time() - start_time)))
    return knn_model

predictions_knn = test_model(train_knn(histogram,train_label),test_data)
predictions_knn = np.reshape(predictions_knn,(1,-1))
accuracy = np.sum(np.array(predictions_knn) == test_label) / float(test_num)
print "The accuracy of my model 2 is {:.2f}%".format(accuracy*100)
 
pred2, label2 = (np.reshape(predictions_knn,(-1)), test_label)

# Training using OneVsRest SVM model

#SVM
def train_svm(X,Y, c):
    start_time = time.time()
    print "Training ..."
    svm_model = OneVsRestClassifier(SVC(kernel = 'linear',C = c)).fit(X,Y)
    print "Training completed"
    print ("Time taken in training - {0}".format((time.time() - start_time)))
    return svm_model

predictions_svm = test_model(train_svm(histogram,train_label,0.8),test_data)
predictions_svm = np.reshape(predictions_svm,(1,-1))
accuracy = np.sum(np.array(predictions_svm) == test_label) / float(test_num)
print "The accuracy of my model 3 is {:.2f}%".format(accuracy*100)
 

pred3, label3 = (np.reshape(predictions_svm,(-1)), test_label) 
