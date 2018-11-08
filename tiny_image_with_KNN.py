# classification using tiny images and KNN

def feature_extract(raw_data):
    feat_dim = 256
    feat = np.zeros((len(raw_data), feat_dim), dtype=np.float32)
    for i in xrange(feat.shape[0]):
        feat[i] = cv2.equalizeHist(cv2.resize(raw_data[i],(16,16))).flatten()
    return feat

def train_KNN(X, Y):
    start_time = time.time()
    clf = KNeighborsClassifier(len(np.unique(Y)))
    clf.fit(X,Y)
    print ("Time taken in training {0}".format((time.time() - start_time)))
    return clf 

def predict_KNN(model, x):
    return model.predict(x)

train_feat = feature_extract(train_data)
test_feat = feature_extract(test_data)

start_time = time.time()
predictions = predict_KNN(train_KNN(train_feat, train_label), test_feat)  
print ("Time taken in testing {0}".format((time.time() - start_time)))
accuracy = sum(np.array(predictions) == test_label) / float(test_num)
print "The accuracy of my model 1 is {:.2f}%".format(accuracy*100)

pred1, label1 = (predictions, test_label)