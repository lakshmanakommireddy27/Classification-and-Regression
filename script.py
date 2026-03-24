
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    data = train_data.shape[0]
    nfeature = train_data.shape[1]

    initialWeights = initialWeights.reshape((nfeature + 1, 1))

    bt = np.ones((data, 1))
    X = np.hstack((bt, train_data))

    z = np.dot(X, initialWeights)
    h = sigmoid(z)

    error = (-1.0 / data) * np.sum(labeli * np.log(h) + (1 - labeli) * np.log(1 - h))

    error_grad = (1.0 / data) * np.dot(X.T, (h - labeli))
    error_grad = error_grad.ravel()

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    ndata = data.shape[0]
    bt = np.ones((ndata, 1))
    X = np.hstack((bt, data))
    p = sigmoid(np.dot(X, W))
    label = np.argmax(p, axis=1).reshape((ndata, 1))
    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    data = train_data.shape[0]
    nfeature = train_data.shape[1]
    nclass = labeli.shape[1]

    params = params.reshape((nfeature + 1, nclass))

    bt = np.ones((data, 1))
    X = np.hstack((bt, train_data))

    s = np.dot(X, params)

    es = np.exp(s - np.max(s, axis=1, keepdims=True))
    probs = es / np.sum(es, axis=1, keepdims=True)

    error = (-1.0 / data) * np.sum(labeli * np.log(probs + 1e-15))

    error_grad = (1.0 / data) * np.dot(X.T, (probs - labeli))
    error_grad = error_grad.ravel()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    ndata = data.shape[0]
    bt = np.ones((ndata, 1))
    X = np.hstack((bt, data))
    s = np.dot(X, W)
    es = np.exp(s - np.max(s, axis=1, keepdims=True))
    p = es / np.sum(es, axis=1, keepdims=True)
    label = np.argmax(p, axis=1).reshape((ndata, 1))
    return label


if __name__ == "__main__":
    """
    Script for Logistic Regression
    """
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    n_class = 10
    n_train = train_data.shape[0]
    n_feature = train_data.shape[1]

    Y = np.zeros((n_train, n_class))
    for i in range(n_class):
        Y[:, i] = (train_label == i).astype(int).ravel()\


    W = np.zeros((n_feature + 1, n_class))
    initialWeights = np.zeros((n_feature + 1,))
    opts = {'maxiter': 100}
    for i in range(n_class):
        labeli = Y[:, i].reshape(n_train, 1)
        args = (train_data, labeli)
        nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        W[:, i] = nn_params.x

    predicted_label = blrPredict(W, train_data)
    print('\n Training set Accuracy for BLR:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = blrPredict(W, validation_data)
    print('\n Validation set Accuracy for BLR:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = blrPredict(W, test_data)
    print('\n Testing set Accuracy for BLR:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
    
    """
    Script for Support Vector Machine
    """

    print('\n\n--------------SVM-------------------\n\n')
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(train_data, train_label.ravel())

    print('\n Training set Accuracy with Linear SVM:' + str(100 * svm_linear.score(train_data, train_label)) + '%')
    print('\n Validation set Accuracy with Linear SVM:' + str(
        100 * svm_linear.score(validation_data, validation_label)) + '%')
    print('\n Testing set Accuracy with Linear SVM:' + str(100 * svm_linear.score(test_data, test_label)) + '%')

    svm_rbf = SVC(kernel='rbf', gamma=1)
    svm_rbf.fit(train_data, train_label.ravel())

    print('\n Training set Accuracy with RBF SVM and gamma = 1:' + str(100 * svm_rbf.score(train_data, train_label)) + '%')
    print('\n Validation set Accuracy with RBF SVM and gamma = 1:' + str(100 * svm_rbf.score(validation_data, validation_label)) + '%')
    print('\n Testing set Accuracy with RBF SVM and gamma = 1:' + str(100 * svm_rbf.score(test_data, test_label)) + '%')

    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(train_data, train_label.ravel())

    print('\n Training set Accuracy with RBF SVM and default parameters:' + str(100 * svm_rbf.score(train_data, train_label)) + '%')
    print('\n Validation set Accuracy RBF SVM and default parameters:' + str(100 * svm_rbf.score(validation_data, validation_label)) + '%')
    print('\n Testing set Accuracy RBF SVM and default parameters:' + str(100 * svm_rbf.score(test_data, test_label)) + '%')

    training_accuracy = np.zeros(11)
    validation_accuracy = np.zeros(11)
    testing_accuracy = np.zeros(11)
    cValues = np.zeros(11)
    cValues[0] = 1.0
    cValues[1:] = [x for x in np.arange(10.0, 101.0, 10.0)]
    for i in range(11):
        svm_rbf = SVC(C=cValues[i],kernel='rbf')
        svm_rbf.fit(train_data, train_label.flatten())
        training_accuracy[i] = 100*svm_rbf.score(train_data, train_label)
        validation_accuracy[i] = 100*svm_rbf.score(validation_data, validation_label)
        testing_accuracy[i] = 100*svm_rbf.score(test_data, test_label)
        print('\n Training set Accuracy with C Value = ' + str(cValues[i]) + str(":::") + str(training_accuracy[i]) + '%')
        print('\n Validation set Accuracy with C Value = ' + str(cValues[i]) + str(":::") + str(validation_accuracy[i]) + '%')
        print('\n Testing set Accuracy with C Value = ' + str(cValues[i]) + str(":::") +str(testing_accuracy[i]) + '%')

    plt.plot(cValues, training_accuracy, 'o-',
        cValues, validation_accuracy,'o-',
        cValues, testing_accuracy, 'o-')

    
    plt.title('SVM with Gaussian kernel for multiple values of C')
    plt.legend(('Train','Validation','Test'), loc='upper left')
    plt.xlabel('C Values')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig("multipleCValuesRBF.png")
    plt.show()

    """
    Script for Extra Credit Part
    """
    # FOR EXTRA CREDIT ONLY
    W_b = np.zeros((n_feature + 1, n_class))
    initialWeights_b = np.zeros((n_feature + 1) * n_class)
    args_b = (train_data, Y)
    opts_b = {'maxiter': 100}
    nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
    W_b = nn_params.x.reshape((n_feature + 1, n_class))

    predicted_label_b = mlrPredict(W_b, train_data)
    print('\n Training set Accuracy for MLR:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

    predicted_label_b = mlrPredict(W_b, validation_data)
    print('\n Validation set Accuracy for MLR:' + str(
        100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

    predicted_label_b = mlrPredict(W_b, test_data)
    print('\n Testing set Accuracy for MLR:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%') 