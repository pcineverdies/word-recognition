## The architecture

The architecture is based on [[1](https://arxiv.org/abs/1610.00087)]. The paper is quite old (2016, which was much time ago for machine learning), but I thing it can be useful to dive into the topic. I started studying machine learning with the "School in AI", and I still have to master some basic concepts of it, such as the convolutional layers. Furthermore, the paper itself says that adding more convolutional layers makes training harder, but it outperforms networks based on spectrogram of audio.

The network we adopted is M5, which is implemented in Keras in [[2](https://github.com/philipperemy/very-deep-convnets-raw-waveforms)].

```python
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
		# x.shape = (N, 1, 8000)
		x = self.conv1(x)
		# x.shape = (N, 32, 496)
		x = self.bn1(x)
		# x.shape = (N, 32, 496)
		x = F.relu(x)
		# x.shape = (N, 32, 496)
		x = self.pool1(x)
		# x.shape = (N, 32, 124)
		x = self.conv2(x)
		# x.shape = (N, 32, 122)
		x = self.bn2(x)
		# x.shape = (N, 32, 122)
		x = F.relu(x)
		# x.shape = (N, 32, 122)
		x = self.pool2(x)
		# x.shape = (N, 32, 30)
		x = self.conv3(x)
		# x.shape = (N, 64, 28)
		x = self.bn3(x)
		# x.shape = (N, 64, 28)
		x = F.relu(x)
		# x.shape = (N, 64, 28)
		x = self.pool3(x)
		# x.shape = (N, 64, 7)
		x = self.conv4(x)
		# x.shape = (N, 64, 5)
		x = self.bn4(x)
		# x.shape = (N, 64, 5)
		x = F.relu(x)
		# x.shape = (N, 64, 5)
		x = self.pool4(x)
		# x.shape = (N, 64, 1)
		x = F.avg_pool1d(x, x.shape[-1])
		# x.shape = (N, 64, 1)
		x = x.permute(0, 2, 1)
		# x.shape = (N, 1, 64)
		x = self.fc1(x)
		# x.shape = (N, 1, 35)
		x = F.log_softmax(x, dim=2)
		# x.shape = (N, 1, 35)
		return x
```

Let's see how each layer changes the dimension of the input. 

- `nn.Conv1d`: Input: $(C_{\text{in}},L_{\text{in}})$; Output: $(C_{\text{out}},L_{\text{out}})$ where 

$$C_{\text{out}} = \text{n\_channel}$$ 

$$L_{\text{out}} = \left[\frac{L_{\text{in}}+ 2 \cdot \text{padding} - \text{dilation} \cdot (\text{kernel\_size}-1) -1}{\text{stride}}+1\right]$$

- `nn.BatchNorm1d`: Output has the same shape as input. 
- `nn.MaxPool1d`: - `nn.Conv1d`: Input: $(C,L_{\text{in}})$; Output: $(C ,L_{\text{out}})$ where 

$$L_{\text{out}} = \left[\frac{L_{\text{in}}- \text{kernel\_size}}{\text{kernel\_size}}+1\right]$$

- `F.log_softmax`: it is equivalent to a softmax followed by a logarithm (`log(softmax(x))`), but numerically more stable. The `dim` attribute tells on which dimension the operation is done. 

The input of the network is a waveform of shape `(1, 8000)` (the first dimension should be the batch size, but we can ignore it). Using the above expressions, we can write the shape of `x` after each layer of the network. The shape of the output is clearly `(1, 35)`, corresponding to the number of different labels.

Using some simple functions, the number of parameters to learn is $26915$. 

References: [[1](https://arxiv.org/abs/1610.00087)], [[2](https://github.com/philipperemy/very-deep-convnets-raw-waveforms)].