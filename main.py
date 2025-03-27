import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from cifar100_dataloader import get_cifar100_dataloader
from res_net_prune import ResNetPruned
import os

if __name__ == '__main__':
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    base_name = '2e3_gamma_5e5_bigger_batch_sgd_mom'
    batch_size = 160
    learning_rate = 2e-3
    num_epochs = 300
    kill_threshold = 1e-4  # Pruning threshold
    use_or_lose = True
    gamma_init = 5e-5
    base_gamma = gamma_init * learning_rate
    adjustment_factor = 0.1
    done_dying = False
    gamma_importance = {}
    # Load data
    train_loader, test_loader = get_cifar100_dataloader(batch_size=batch_size)

    # Initialize model, loss, optimizer
    model = ResNetPruned(num_classes=100, kill_threshold=kill_threshold).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, min_lr=5e-6, factor=0.5)
    # Set up TensorBoard
    log_dir = 'loggin_stuff'
    writer = SummaryWriter(log_dir=os.path.join(log_dir, base_name), flush_secs=1)


    def evaluate(model, test_loader, criterion, epoch):
        """Evaluate model on test set and log test loss and accuracy."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total

        writer.add_scalar("Test Loss", avg_loss, epoch)
        writer.add_scalar("Test Accuracy", accuracy, epoch)

        tot_params, tot_used_params = count_params_with_masking(model)
        writer.add_scalar("Used Parameters", tot_used_params, epoch)
        print(f"[Epoch {epoch + 1}] Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Total Params: {tot_params:.2f} Total Used Params: {tot_used_params:.2f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]}")
        return avg_loss, accuracy, tot_used_params


    # Training loop
    print("Starting training...")
    global_step = 0
    greedy_loss = 4.605


    def get_layer_importance(this_param, mask):
        # Apply mask to exclude killed nodes
        active_weights = this_param.data * mask  # Zero out killed nodes

        # Compute mean and variance only over active nodes
        active_mean = active_weights[mask.bool()].mean().item()  # Avoid counting zeroed-out weights
        activation_variance = ((active_weights[mask.bool()] - active_mean).pow(2)).mean().item()

        mean_abs = active_weights.abs().mean().item() + 1e-6  # Ensure no division by zero

        return mean_abs / activation_variance


    def count_params_with_masking(model):
        total_params = 0
        total_unmasked_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()  # Count all parameters
            clean_name = name.replace(".", "_").replace("_weight", "")
            if 'weight' in name and len(param.shape) > 1:
                mask = model.masks[clean_name]
                unmasked_params = mask.sum().item()

                total_unmasked_params += unmasked_params
            else:
                # If it's not a masked layer (biases, BatchNorm, etc.), count as fully unmasked
                total_unmasked_params += param.numel()

        return total_params, total_unmasked_params


    for epoch in range(num_epochs):
        model.train()
        batch_loss = 0.0
        # update gamma_importance with killed nodes and possibly new base_gamma
        for name, param in model.named_parameters():
            clean_name = name.replace(".", "_").replace("_weight", "")
            if 'weight' in name and len(param.shape) > 1:
                mask = model.masks.get(clean_name)
                importance = get_layer_importance(param, mask)
                gamma_importance[clean_name] = torch.full_like(param, (importance * base_gamma))
                writer.add_scalar(f"ActivationImportance/{name}", importance, global_step)

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            batch_loss += loss.item()

            # Backward pass
            loss.backward()

            if use_or_lose:
                for name, layer in model.named_modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        clean_name = name.replace(".", "_")
                        mask = model.masks[clean_name]
                        if layer.weight.grad is not None:
                            layer.weight.grad *= mask  # Zero out gradients for pruned weights

            optimizer.step()

            if use_or_lose and epoch > 1 and not done_dying:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        clean_name = name.replace(".", "_").replace("_weight", "")
                        if 'weight' in name and len(param.shape) > 1:
                            mask = model.masks.get(clean_name)
                            gamma = torch.minimum(gamma_importance[clean_name], param.data.abs())
                            param.data -= gamma * param.data.sign() * mask

            # Log batch loss
            # Print update every 10 batches
            if (batch_idx + 1) % 10 == 0:
                # batch loss
                writer.add_scalar("Train Loss", batch_loss / 10, global_step)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {batch_loss / 10:.4f}")

                batch_loss = 0.0

                # Log layer-wise weights and gradient updates
                for name, param in model.named_parameters():
                    if 'weight' in name and len(param.shape) > 1:
                        avg_grad_update = (param.grad.abs()).mean().item()
                        writer.add_scalar(f"GradUpdates/{name}", avg_grad_update, global_step)

            global_step += 1  # Increment global step for TensorBoard logging

        # Evaluate after each epoch
        test_loss, test_accuracy, num_params_used = evaluate(model, test_loader, criterion, epoch)

        if num_params_used < 550_000:
            done_dying = True
            print("done dying!")
        # Step scheduler (maybe decrease lr)
        if done_dying:
            scheduler.step(test_loss)

        # Kill small weights
        if use_or_lose:
            model.kill_small_weights(writer=writer, global_step=global_step)

        # update base gamma
        base_gamma = gamma_init * optimizer.param_groups[0]["lr"]

    writer.close()
    print("Training complete! Logs saved to:", log_dir)
