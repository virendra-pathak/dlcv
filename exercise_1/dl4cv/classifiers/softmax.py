import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    
    # each i is a single image in 1D
    for i in range(num_train):
        # This represent vector of scores for each classes, similar multiclass SVM
        f_i = X[i].dot(W)

        # for numerical stabilty, as exp can grow into a very large number.
        f_i -= np.max(f_i)

        # keep adding loss for each images(or sample)
        sum_j = np.sum(np.exp(f_i))
        # softmax take a real number 'a' and transform into between (0,1), similar to probability
        # prob is a 1D vector of dimension number of classes
        prob = np.exp(f_i)/sum_j
        loss += -np.log(prob[y[i]]) 

        # Compute gradient
        # Here we are computing the contribution to the inner sum for a given i.
        for k in range(num_classes):
            dW[:, k] += (prob[k] - (k == y[i])) * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    #f = X.dot(W)
    f = np.dot(X,W)
    f -= np.max(f, axis=1, keepdims=True) # max of every sample
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f)/sum_f
    #----------experiment-----------------------
    #print("VKP experiment - Start")
    #print("p shape = ", p.shape)
    #print("p[] shape", p[np.arange(num_train), y].shape)
    #print("VKP experiment - End")
    #-------------------------------------------
    
    prob_of_right_class = p[np.arange(num_train), y]
    loss = np.sum(-np.log(prob_of_right_class))
    
    ind = np.zeros_like(p)
    ind[np.arange(num_train), y] = 1
    p = p - ind
    dW = X.T.dot(p)
    
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW