#### 2026-11-12
I hold out 5,000 training images for validation and train on the remaining 55,000; the paper trains on all 60,000.

#### 2026-11-13
MNIST is a set of 28 by 28 grayscale images (60,000 images), each image paired with its label. Pixel brightnesses are normalized. 784 pixels in an image represent 784 neurons of the input layer.

A neuron $z = w \cdot x + b = w_1x_1 + w_2x_2 + ... + w_{784}x_{784} + b.$ A layer is many neurons at once. In code, `load()` returns `X` that has shape `(batch, 784)`​. `X`​ stores one image per row. `x @ W + b` equals

$$(Wx + b)^T = x^T W^T + b^T$$

Stacked layers are the same as one layer. Let layers $h = 2x + 1$ and $z = 3h + 4$. 

$$
\begin{aligned}
z &= 3h + 4 \\
&= 3(2x + 1) + 4 \\
&= 6x + 3 + 4 \\
&= 6x + 7
\end{aligned}
$$

$h+z$ becomes just one layer with weight 6 and bias 7.

Putting a non-linear function between the layers ensures no single layer can reproduce the stack: the model now behaves in a way one layer cannot.

MLP is a network where every neuron in one layer connects to every neuron in the next. A fully connected network with fully connected layers (dense layers). A 28 by 28 image is flattened into a long vector of 784 numbers to feed into an MLP. After flattening, pixels are unrelated to each other, and the network cannot recognize the neighbors.

The hidden layers just describe the middle layers and nothing more.

I can adjust 3 dials to build a shape-checking MLP. Dial 1: how many hidden layers, dial 2: how many hidden neurons, and dial 3: activation function. The weights are random. (Input layer and output layer are decided by the data!)

1. how many hidden layers: One hidden layer + ReLU is the minimum structure that makes the model nonlinear. 
2. Assume the reasonably matched baseline is 62,000 parameters, $784h + h + 10h + 10 = 795h + 10 ~= 62000$. So there are 78 hidden neurons after the parameter matching. First add 128 as a placeholder.
3. ReLU is placed after hidden layers. The original paper used tanh.

```
(batch, 28, 28) --> flatten --> (batch, 784) --> dense --> ReLU --> (batch, 128) --> dense --> (batch, 10)
```
