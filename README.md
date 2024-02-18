# PyTorch CNN Tutorial for MNIST Digit Classification

This is a PyTorch CNN tutorial for MNIST digit classification. The tutorial is divided into the following sections:
- [Introduction to PyTorch](#introduction-to-pytorch)
- [Introduction to MNIST Dataset](#introduction-to-mnist-dataset)
- [Define the CNN Model](#define-the-cnn-model)
- [Prepare the Dataset](#prepare-the-dataset)
- [Train the Model](#train-the-model)
- [Evaluate the Model](#evaluate-the-model)
- [Summary](#summary)

## Introduction to PyTorch

PyTorch is an open source machine learning library based on the Torch library. It is used for applications such as natural language processing. It is primarily developed by Facebook's AI Research lab. PyTorch provides two high-level features:
- Tensor computation (like NumPy) with strong GPU acceleration
- Deep neural networks built on a tape-based autograd system

## Introduction to MNIST Dataset

The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. The MNIST database contains 60,000 training images and 10,000 testing images. The images are grayscale, 28x28 pixels, and centered to reduce preprocessing and get started quicker.

## Define the CNN Model

We will define a simple CNN model for digit classification. The model will have the following layers:

### Convolutional Layers:

`conv1`: The first convolutional layer takes a single-channel (grayscale) image as input (in_channels=1) and produces 32 feature maps (out_channels=32) with a kernel size of 3x3. This layer aims to capture low-level features such as edges and simple textures. The effective receptive field after this layer is 3x3 pixels, with the output size remaining at 28x28 due to the absence of padding and stride set to 1.

`conv2`: The second convolutional layer takes the 32 feature maps as input and produces 64 feature maps with a kernel size of 3x3. This layer further processes the features extracted by conv1, capturing more complex patterns. Following this layer, a max pooling operation with a 2x2 window and stride of 2 halves the feature map dimensions to 13x13 and increases the effective receptive field.

`conv3`: This layer increases the depth to 128 feature maps with the same kernel size of 3x3, processing the pooled feature maps from conv2. It does not include a pooling step, so the spatial dimensions reduce slightly due to the convolution operation.

`conv4`: The fourth convolutional layer expands the feature maps to 256 using the same kernel size. It is followed by another max pooling operation, which further reduces the spatial dimensions and increases the receptive field. This layer helps the network to capture high-level features in the image.

### Fully Connected Layers:

After the series of convolutions and pooling, the feature maps are flattened into a single vector of size 4096 (256 feature maps * 4 * 4 spatial dimensions) to feed into fully connected layers.
`fc1`: The first fully connected layer reduces the dimension from 4096 to 50, allowing the network to learn a non-linear combination of the high-level features extracted by the convolutional layers.
`fc2`: The final fully connected layer further reduces the dimension from 50 to 10, corresponding to the 10 classes of digits (0-9) in the MNIST dataset.

### Activation Functions and Pooling:

The network uses ReLU activation functions after each convolutional layer to introduce non-linearity, allowing it to learn complex patterns in the data.
Max pooling operations are used after conv2 and conv4 to reduce the spatial dimensions of the feature maps, effectively increasing the receptive field and reducing the computation for subsequent layers.

### Output:

The output of the network is passed through a log softmax function, which is commonly used for multi-class classification problems. It provides a probability distribution over the 10 digit classes for each input image.

## Prepare the Dataset

We will prepare the MNIST dataset by downloading the images and creating a PyTorch dataset using the `torchvision` library. We will also define a data loader to load the data in batches during training and evaluation.

```python
# MNIST Training dataset with specified transformation
train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
# MNIST Testing dataset with specified transformation
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
# Set the batch size to 512, indicating the number of samples to be processed in one go.
batch_size = 512

# Define a dictionary of keyword arguments for the DataLoader:
# 'batch_size': Specifies the number of samples in each batch.
# 'shuffle': If True, the dataset will be shuffled at the beginning of each epoch to reduce model overfitting.
# 'num_workers': Sets the number of subprocesses to use for data loading. Utilizing multiple workers can enhance data loading throughput.
# 'pin_memory': When set to True and using a CUDA-enabled GPU, this option pins memory, potentially speeding up data transfer to the GPU.
kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

# Initialize the DataLoader for the test dataset.
# 'test_data': The dataset to use for testing.
# The DataLoader handles efficient loading of data from 'test_data' using the parameters defined in 'kwargs'.
test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

# Initialize the DataLoader for the training dataset.
# 'train_data': The dataset to use for training. The DataLoader will shuffle this data if 'shuffle' is True, as per 'kwargs'.
# This DataLoader facilitates efficient loading of training data, respecting the parameters specified in 'kwargs'.
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
```

### Data Augmentation:

#### Training Dataset Transformations

`transforms.RandomApply`: Applies a list of transformations (in this case, center cropping the image to 22x22 pixels) with a certain probability (p=0.1). This introduces variability in the dataset by randomly cropping images, which helps the model learn to recognize digits regardless of their position in the image.

`transforms.Resize`: Resizes the image to 28x28 pixels, ensuring that all images fed into the model have a uniform size. This is essential since the input layer of neural networks expects a fixed-size input.

`transforms.RandomRotation`: Randomly rotates the image by a degree chosen uniformly from the range [-15, 15] degrees, filling the remaining areas with black (0). This introduces rotational variance, helping the model to recognize digits regardless of their orientation.

`transforms.ToTensor`: Converts the image to a PyTorch tensor, which is the required input format for PyTorch models. This also scales the pixel values to the range [0, 1].

`transforms.Normalize`: Normalizes the tensor image with mean and standard deviation. This step is crucial for converging faster during training, as it ensures that the input features are on a similar scale. The mean (0.1307) and standard deviation (0.3081) values are typically chosen based on the dataset's global mean and standard deviation.

#### Testing Dataset Transformations

`transforms.ToTensor` and `transforms.Normalize` are applied to the testing dataset without the data augmentation steps (like random cropping and rotation). This is because, during testing, we evaluate the model's performance on unaltered images to simulate real-world application. The normalization parameters (mean: 0.1407, standard deviation: 0.4081) for the testing set might be slightly different, which could be an oversight or intentionally adjusted based on the testing set distribution. However, it's generally recommended to use the same normalization parameters for both training and testing datasets for consistency.

## Train the Model

We will train the CNN model using the training dataset. The training process involves the following steps:
- Forward pass: The input image is fed into the Convolutional Neural Network model, and the predicted output is computed.
- Loss computation: The difference between the predicted output and the actual target is computed using the negative log-likelihood loss function.
- Backward pass: The gradients of the loss with respect to the model parameters are computed using backpropagation.
- Optimization: The model parameters are updated using Stochastic Gradient Descent to minimize the loss.

## Evaluate the Model

We will evaluate the trained model using the testing dataset. The evaluation process involves the following steps:
- Forward pass: The input image is fed into the trained model, and the predicted output is computed.
- Prediction: The class with the highest probability in the predicted output is chosen as the final prediction.
- Accuracy computation: The accuracy of the model is computed by comparing the predicted labels with the actual labels in the testing dataset.

## Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

The CNN model consists of 593,200 parameters, which are learned during the training process. The model is trained using the MNIST dataset and achieves an accuracy of 99.34% on the testing dataset.

## Loss and Accuracy Plots

```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.8040 Batch_id=117 Accuracy=32.28: 100%|██████████| 118/118 [00:30<00:00,  3.91it/s]
Test set: Average loss: 0.0017, Accuracy: 7098/10000 (70.98%)
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.1180 Batch_id=117 Accuracy=88.46: 100%|██████████| 118/118 [00:20<00:00,  5.62it/s]
Test set: Average loss: 0.0003, Accuracy: 9655/10000 (96.55%)
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.1310 Batch_id=117 Accuracy=95.69: 100%|██████████| 118/118 [00:19<00:00,  5.94it/s]
Test set: Average loss: 0.0002, Accuracy: 9776/10000 (97.76%)
Adjusting learning rate of group 0 to 1.0000e-02.

......
......
......

Epoch 18
Train: Loss=0.0050 Batch_id=117 Accuracy=99.10: 100%|██████████| 118/118 [00:21<00:00,  5.55it/s]
Test set: Average loss: 0.0000, Accuracy: 9933/10000 (99.33%)
Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.0343 Batch_id=117 Accuracy=99.08: 100%|██████████| 118/118 [00:21<00:00,  5.53it/s]
Test set: Average loss: 0.0000, Accuracy: 9932/10000 (99.32%)
Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0285 Batch_id=117 Accuracy=99.05: 100%|██████████| 118/118 [00:21<00:00,  5.55it/s]
Test set: Average loss: 0.0000, Accuracy: 9934/10000 (99.34%)
Adjusting learning rate of group 0 to 1.0000e-03.
```


