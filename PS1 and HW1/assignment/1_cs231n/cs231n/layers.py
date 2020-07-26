import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # pass
  new_x = np.reshape(x, (x.shape[0], w.shape[0]))
  out = new_x.dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # pass
  # input_dim = x.shape
  # element_size = np.prod(input_dim[1:])
  new_x = np.reshape(x, (x.shape[0], w.shape[0]))
  temp = dout.dot(w.T)
  dw = new_x.T.dot(dout)
  db = np.sum(dout,axis=0)
  dx = dout.dot(w.T).reshape(x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # pass
  out = np.maximum(x,0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # pass
  temp = (x>0)
  dx = dout * temp
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # pass
  def conv_step(x,w,b):
    s = np.multiply(x,w)
    Z = np.sum(s)
    Z = Z + b
    return Z


  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  conv_padding = ((0,),(0,),(pad,),(pad,))
  padded_x = np.pad(x,conv_padding,'constant')

  h_out = int(1 + (H + 2*pad - HH)/stride)
  row_range = H + 2 * pad - HH + 1
  w_out = int(1 + (W + 2*pad - WW)/stride)
  col_range = W + 2 * pad - WW + 1

  #initialize output volume
  out = np.zeros((N, F, h_out, w_out), dtype=x.dtype)

  for i in range(N):
    for j in range(F):
      for k in range(h_out):
        top = k*stride
        bottom = top + HH
        for l in range(w_out):
          left = l*stride
          right = left + WW
          x_temp = padded_x[i,:, top : bottom, left : right]
          w_temp = w[j]
          b_temp = b[j]
          out[i, j, k, l] = conv_step(x_temp,w_temp,b_temp)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # pass
  def conv_step(x,w,b):
    s = np.multiply(x,w)
    Z = np.sum(s)
    Z = Z + b
    return Z

  x,w,b,conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']

  db = np.sum(dout, axis=(0,2,3))
  dx = np.zeros(x.shape)
  dw = np.zeros(w.shape)

  N, C, H_old, W_old = x.shape
  F, C, HH, WW = w.shape
  N, F, h_out, w_out = dout.shape

  # print(x.shape,w.shape,dout.shape)
  conv_padding = ((0,),(0,),(pad,),(pad,))
  padded_x = np.pad(x,conv_padding,'constant')
  padded_dx = np.zeros_like(padded_x)

  for i in range(N):
    for j in range(F):
      for k in range(h_out):
        top = k*stride
        bottom = top + HH
        for l in range(w_out):
          left = l*stride
          right = left + WW
          x_temp = dout[i,j,k,l]
          w_temp = w[j,:,:,:]
          # b_temp = b[j]
          padded_dx[i,:,top:bottom,left:right] += x_temp*w_temp
          dw[j,:,:,:] += x_temp*padded_x[i,:,top:bottom,left:right]

  #extract dx from dx_padded
  dx = padded_dx[:, :, pad:pad+H_old, pad:pad+W_old]


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # pass
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  stride = pool_param['stride']
  N,C,H,W = x.shape

  H_out = int(1 + (H-ph)/stride)
  W_out = int(1 + (W-pw)/stride)

  out = np.zeros((N, C, H_out, W_out))

  for i in range(N):
    for j in range(C):
      for k in range(H_out):
        top = k*stride
        bottom = top + ph
        for l in range(W_out):
          left = l*stride
          right = left + pw
          out[i,j,k,l] = np.max(x[i, j, top:bottom, left:right])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # pass
  x,pool_param = cache
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  stride = pool_param['stride']
  N,C,H,W = x.shape

  H_out = int(1 + (H-ph)/stride)
  W_out = int(1 + (W-pw)/stride)

  dx = np.zeros(x.shape)

  for i in range(N):
    for j in range(C):
      for k in range(H_out):
        top = k*stride
        bottom = top + ph
        for l in range(W_out):
          left = l*stride
          right = left + pw
          x_temp = x[i,j,top:bottom,left:right]
          imax = np.max(x_temp)
          x_max = x_temp == imax
          dx[i,j,top:bottom,left:right] += dout[i,j,k,l]*x_max
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

