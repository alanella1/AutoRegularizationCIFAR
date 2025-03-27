# AutoRegularizationCIFAR

**AutoRegularizationCIFAR** employs an evolutionary approach to selectively remove weights from an image classifier model trained on the CIFAR-100 dataset. This method aims to enhance model performance by identifying and pruning unnecessary weights, leading to a more efficient and potentially more accurate model.

## Introduction

In deep learning, model regularization is crucial for preventing overfitting and improving generalization. Traditional methods often involve manual tuning of hyperparameters or applying generic regularization techniques. **AutoRegularizationCIFAR** introduces an evolutionary algorithm to automate the process of weight pruning in a ResNet model trained on the CIFAR-100 dataset. By iteratively selecting and removing less significant weights, the model becomes more efficient without compromising, and potentially enhancing, its performance.

## Project Structure

The repository contains the following key files:

- `cifar100_dataloader.py`: Handles the loading and preprocessing of the CIFAR-100 dataset.
- `main.py`: The main script to initiate the training, pruning, and evaluation processes.
- `regular_model.py`: Defines the architecture of the ResNet model used for classification.
- `res_net_prune.py`: Implements the evolutionary algorithm for weight pruning in the ResNet model.
