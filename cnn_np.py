import numpy as numpy

# GAVE UP AFTER INIT D:

### input volume descriptors
# in_channels = dimension of feature (image = RBG (3))
# height/width of input images

### convolutional layer descriptor 
# num_filters = filters/convolution mapping features -> # sets of weights CNN needs to learn
# filter_size = size of filter matrix (filter_size, filter_size, in_channels) - allocating memory for weights

### connected layer descriptor
# num_classes = classification output
# weight_scale = init to random,small  to ensure neurons learn diff things
def init_cnn_params(in_channels, height, width, num_filters, filter_size, num_classes, weight_scale=1e-2):
    params = {
        # init learnable weights
        "W_conv": weight_scale + np.random.randn(num_filters, in_channels, filter_size, filter_size),
        # init learnable biases
        "b_conv": np.zeros(num_filters, 1)), 
    }
    conv_out_h = height - filter_size + 1
    conv_out_w = width - filter_size + 1
    params["W_fc"] = weight_scale + np.random.randn(num_filters * conv_out_h * conv_out_w, num_classes)
    params["b_fc"] = np.zeros((num_classes, ))
    return params, conv_out_h, conv_out_w

# math operation for convolutional layer (forward ops Wx + b)
# loop through all batches, filters, convolution window
# takes region at that window, compress into matrix (patch of RBG values)
# dot product against weights, add bias. 
def conv2d(x, w, b, stride=1): 
    # (input data) N = batch size, C_in = input channels, H = input height, W = input width
    N, C_in, H, W = x.shape
    # (param) F = num filters, HH = filter height, WW = filter width
    F, _, HH, WW = w.shape
    H_out = (H - HH) // stride + 1
    W_out = (W - WW) // stride + 1
    out = np.zeros((N, F, H_out, W_out))
    # calc result for every iamge in batch and for every filter in filter bank
    # batch size
    for n in range(N):
        # number of filters
        for f in range(F):
            # sliding window
            for i in range(H_out): 
                h_start = i * stride
                for j in range(W_out):
                    w_start = j * stride
                    region = x[n, :, h_start:h_start + HH, w_start:w_start + WW]
                    out[n, f, i, j] = np.sum(region * w[f]) + b[f]

   
                for j in range(W_out):
                    w_start = j * stride
                    region = x[n, :, h_start:h_start + HH, w_start:w_start + WW]
                    out[n, f, i, j] = np.sum(region * w[f]) + b[f]
    return out

def conv2d_backward(grad_out, x, w, stride=1):
    N, C_in, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = grad_out.shape
    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)
    # sum error signal across spatial (H, W) and batch (N)
    grad_b = grad_out.sum(axis=(0, 2, 3), keepdims=True)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                h_start = i * stride
                for j in range(W_out):
                    w_start = j * stride
                    region = x[n, :, h_start:h_start + HH, w_start:w_start + WW]
                    # gradient distribution
                    # error signal = derivative of loss w.r.t. output layer
                    grad_w[f] += grad_out[n, f, i, j] * region # gradient of param
                    # find gradinte of input
                    grad_x[n, :, h_start:h_start + HH, w_start:w_start + WW] += grad_out[n, f, i, j] * w[f]
    return grad_x, grad_w, grad_b

# rectified linear unit
def relu(x):
    return np.maximum(x, 0), x > 0

# error signal * deriv of ReLu (error gating/relay)
def relu_backward(grad_out, mask):
    return grad_out * mask

def linear_forward(x, w, b):
    return x @ w + b

# backprop for Fully Connected Layer (last) -> deal w/ 2d matrices
def linear_backward(grad_out, x, w):
    grad_x = grad_out @ w.t
    grad_w = x.T @ grad_out
    grad_b = grad_out.sum(axis=0)
    return grad_x, grad_w, grad_b

def softmax_cross_entropy(logits, labels):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(logits)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), labels]).mean()
    grad = probs
    grad[np.arange(N), labels] -= 1
    grad /= N
    return loss, grad

def forward(params, x):
    conv_out = conv2d(x, params["W_conv"], params["b_conv"])
    relu_out, relu_mask = relu(conv_out)
    flat = relu_out.reshape(relu_out.shape[0], -1)
    logits = linear_forward(flat, params["W_fc"], params["b_fc"])
    cache = {"x": x, "conv_out": conv_out, "relu_mask": relu_mask, "flat": flat, "relu_shape": relu_out.shape}
    return logits, cache
    

def forward(params, x):
    conv_out = conv2d(x, params["W_conv"], params["b_conv"])
    relu_out, relu_mask = relu(conv_out)
    flat = relu_out.reshape(relu_out.shape[0], -1)
    # logits = output of FC operations
    # classificaiton = softmax to logits
    logits = linear_forward(flat, params["W_fc"], params["b_fc"])
    cache = {"x": x, "conv_out": conv_out, "relu_mask": relu_mask, "flat": flat, "relu_shape": relu_out.shape}
    return logits, cache

def backward(params, grad_logits, cache):
    flat = cache["flat"]
    # receive first error signal from loss function
    grad_flat, grad_W_fc, grad_b_fc = linear_backward(grad_logits, flat, params["W_fc"])
    grad_relu = grad_flat.reshape(cache["relu_shape"]) # unflatten 2D -> 4D
    grad_conv = relu_backward(grad_relu, cache["relu_mask"])
    grad_x, grad_W_conv, grad_b_conv = conv2d_backward(grad_conv, cache["x"], params["W_conv"])
    grads = {"W_conv": grad_W_conv, "b_conv": grad_b_conv.squeeze(-1), "W_fc": grad_W_fc, "b_fc": grad_b_fc}
    return grads, grad_x

# optimizer: param_new = param_old - learning_rate * gradient
def sgd_step(params, grads, lr):
    params["W_conv"] -= lr * grads["W_conv"]
    params["b_conv"] -= lr * grads["b_conv"][:, None]
    params["W_fc"] -= lr * grads["W_fc"]
    params["b_fc"] -= lr * grads["b_fc"]


def train_step(params, x, labels, lr):
    logits, cache = forward(params, x)
    # calc loss from softmax
    loss, grad_logits = softmax_cross_entropy(logits, labels)
    grads, _ = backward(params, grad_logits, cache)
    # update param 
    sgd_step(params, grads, lr)
    return loss


if __name__ == "__main__":
    np.random.seed(0)
    batch_size, in_channels, height, width = 4, 1, 28, 28
    num_filters, filter_size, num_classes = 8, 3, 10

    params, _, _ = init_cnn_params(in_channels, height, width, num_filters, filter_size, num_classes)
    x_batch = np.random.randn(batch_size, in_channels, height, width)
    y_batch = np.random.randint(0, num_classes, size=batch_size)

    loss = train_step(params, x_batch, y_batch, lr=1e-2)
    print(f"loss: {loss:.4f}")