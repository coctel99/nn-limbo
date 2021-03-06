import numpy as np

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.linalg.norm(W) ** 2
    grad = reg_strength * 2 * W
    return loss, grad


def softmax(predictions):
    if predictions.ndim == 1:
        predictions_exp = np.exp(predictions - np.max(predictions))
        return predictions_exp / np.sum(predictions_exp)
    else:
        predictions_exp = np.exp(predictions - np.max(predictions, axis=1).reshape(-1, 1))
    soft_max = predictions_exp / np.sum(predictions_exp, axis=1).reshape(-1, 1)

    return soft_max


def cross_entropy_loss(probs, target_index):
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        loss = np.mean(-np.log(probs[np.arange(probs.shape[0]), target_index]))

    return loss


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment

    soft_max = softmax(preds)
    loss = cross_entropy_loss(soft_max, target_index)
    if preds.ndim == 1:
        soft_max[target_index] -= 1
    else:
        soft_max[np.arange(soft_max.shape[0]), target_index] -= 1
        soft_max /= soft_max.shape[0]
    d_preds = soft_max

    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X

        return np.maximum(0, X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        relu_grad = self.X >= 0
        d_result = relu_grad * d_out

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X

        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += d_out.mean(axis=0) * self.X.shape[0]
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 0
        out_width = 0
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        X_padded = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        X_padded[:, self.padding:self.padding + height, self.padding:self.padding + width, :] = X

        self.X_cache = (X, X_padded)

        X_padded = X_padded[:, :, :, :, np.newaxis]
        W = self.W.value[np.newaxis, :, :, :, :]

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                curr_X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, :]
                out[:, y, x, :] = np.sum(curr_X_slice * self.W.value, axis=(1, 2, 3)) + self.B.value

        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        X, X_padded = self.X_cache
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        X_grad = np.zeros_like(X_padded)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                curr_X_slice = X_padded[:, y:y + self.filter_size, x:x + self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(grad * curr_X_slice, axis=0)
                X_grad[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))
        d_input = X_grad[:, self.padding:self.padding + height, self.padding:self.padding + width, :]
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy()

        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        out = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                curr_X_slice = X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                out[:, y, x, :] = np.amax(curr_X_slice, axis=(1, 2))

        return out


    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        out = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                # выделяем текущий слайс по X
                curr_X_slice = self.X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                # получаем значения градиентов
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                # определяем маску
                mask = (curr_X_slice == np.amax(curr_X_slice, (1, 2))[:, np.newaxis, np.newaxis, :])
                # накладываем маску на значения градиентов
                out[:, y:y + self.pool_size, x:x + self.pool_size, :] += grad * mask

        return out

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        result = X.reshape(batch_size, height * width * channels)

        return result

    def backward(self, d_out):
        # TODO: Implement backward pass
        result = d_out.reshape(self.X_shape)
        return result

    def params(self):
        # No params!
        return {}
