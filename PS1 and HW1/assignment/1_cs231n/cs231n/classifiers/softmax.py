import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  # pass
  # Z = Wx + b
  Z = W.dot(X)
  Z = Z - np.max(Z, axis=0)
  N = len(y)
  #P softmax
  P_i = np.exp(Z) 
  # P_i = P_i /  np.sum(np.exp(Z), axis=0)

  #Loss
  C = W.shape[0]
  one_hot = np.zeros((C,N))
  one_hot[y,np.arange(N)] = 1 
  # print(one_hot.shape)
  # print(one_hot.shape, P_i.shape, X.shape)
  # loss -= (1.0/N)*np.sum(np.log(P_i[y, np.arange(N)])) 
  loss -= (1.0/N)*(np.sum(Z[y,np.arange(N)]) - np.sum(np.log(np.sum(np.exp(Z),axis=0))))
  loss += reg*np.sum(W**2)
  #gradient
  dW = 1.0/N*((P_i/np.sum(np.exp(Z), axis=0) - one_hot).dot(X.T)) + 2*reg*W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
