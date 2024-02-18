# Importing packages & libraries from PyTorch
import torch


# Importing tqdm for progress bar visualization during loops
from tqdm import tqdm

# Importing matplotlib for plotting graphs and images
import matplotlib.pyplot as plt


# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

# Function to calculate the number of correct predictions
def GetCorrectPredCount(pPrediction, pLabels):
    # Compares the predicted labels with the actual labels
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

# Function to train the model
def train(model, device, train_loader, optimizer, criterion):
    model.train()  # Sets the model to training mode
    pbar = tqdm(train_loader)  # Initializes a progress bar

    train_loss = 0
    correct = 0
    processed = 0

    # Loop through each batch from the training data
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)  # Move data to the specified device (CPU/GPU)
        optimizer.zero_grad()  # Clears the gradients of all optimized tensors

        # Forward pass: compute predicted outputs by passing inputs to the model
        pred = model(data)

        # Calculate the loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        optimizer.step()  # Performs a single optimization step

        correct += GetCorrectPredCount(pred, target)  # Update total number of correct predictions
        processed += len(data)  # Update total number of processed samples

        # Update progress bar description with current loss and accuracy
        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    # Append average accuracy and loss for the epoch to the lists
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

# Function to evaluate the model on the test dataset
def test(model, device, test_loader, criterion):
    model.eval()  # Sets the model to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Disables gradient calculation to save memory and computations
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss

            correct += GetCorrectPredCount(output, target)  # Update total number of correct predictions

    # Calculate average loss and accuracy
    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    # Print test set results
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')
    
    
    
def plot_accuracy_loss_graphs():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
    