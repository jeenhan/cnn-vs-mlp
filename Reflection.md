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
#### 2026-11-14
$$
x_1 =
\begin{bmatrix} 
\text{pixel 1 in image 1} \\
p_1^2 \\
\vdots \\
p_1^{784}
\end{bmatrix}
$$

$$
X = 
\begin{bmatrix}
x_1^T \\
x_2^T \\
\vdots \\
x_{64}^T
\end{bmatrix}
=
\begin{bmatrix}
\text{pixel 1 in image 1} & p_1^2 & ... & p_1^{784} \\
\text{pixel 1 in image 2} & p_2^2 & ... & p_2^{784} \\
\vdots & \vdots & \vdots & \vdots \\
\text{pixel 1 in image 64} & p_{64}^2 & ... & p_{64}^{784} \\
\end{bmatrix}
$$

Name the next layer's neurons are $a_1, a_2, \dots, a_{128}$.
$$
w_{a_1} =
\begin{bmatrix} 
\text{weight 1 to } a_1 \\
w_{a_1}^2 \\
\vdots \\
w_{a_1}^{784}
\end{bmatrix}
$$

$$
W = 
\begin{bmatrix}
w_{a_1} & w_{a_2} & ... & w_{a_{128}}
\end{bmatrix}
=
\begin{bmatrix}
w_{a_1}^1 & w_{a_2}^1 & ... & w_{a_{128}}^1 \\
w_{a_1}^2 & w_{a_2}^2 & ... & w_{a_{128}}^2 \\
\vdots & \vdots & \vdots & \vdots \\
w_{a_1}^{784} & w_{a_2}^{784} & ... & w_{a_{128}}^{784}
\end{bmatrix}
$$

`h = X @ W + b`:
$$
h =
\begin{bmatrix}
\text{128 activations from the first batch} \\
\text{128 activations from the second batch} \\
\vdots \\
\text{128 activations from the last batch} \\
\end{bmatrix}
$$

Starting weights at 0 will always compute the same output. Starting at random values will grow too large when adding 784 of them. Initialize weights using He initialization. *He initialization draws x from the normal distribution and divide by $\sqrt{n}$. Why?

#### 2026-11-21
Each image has 10 scores like `1.2` or `-0.4`. The biggest score is the prediction. Softmax is the function that converts a score to probability. $e^{\text{some score}}$ has 2 properties:
1. $e^{\text{some score}}$ is always positive, and
2. a bigger score gives a bigger $e^{\text{some score}}$.

Then divide each by total: 
$$
p_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + \dots + e^{z_{10}}} =
\frac{e^{z_1}}{\sum_{k=1}^{10} e^{z_k}}
$$

Subtract the biggest score from all scores to prevent the overflow:
$$(1000, 999, 998) - 1000 = (0, -1, -2)$$
