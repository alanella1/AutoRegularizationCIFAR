import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block with two conv layers"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (identity or downsampling)
        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:  # Only create a shortcut if needed
            layers = []
            if stride != 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsampling first if needed
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))  # Channel matching
            layers.append(nn.BatchNorm2d(out_channels))

            self.shortcut = nn.Sequential(*layers)  # Convert to sequential block

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add residual connection
        return F.relu(out)


class ResNetPruned(nn.Module):
    """ResNet-18 with Built-in Weight Pruning"""

    def __init__(self, num_classes=100, kill_threshold=1e-3):
        super(ResNetPruned, self).__init__()

        self.kill_threshold = kill_threshold  # Pruning threshold

        # Two-Path Stem
        self.local_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.global_conv = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.merge_conv = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2)
        # self.reduced_layer3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(512)
        self.layer3 = self._make_layer(256, 512, num_blocks=1, stride=2)
        # self.layer4 = self._make_layer(512, 512, num_blocks=2, stride=2)

        # Global Average Pooling + Fully Connected Layer
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(512, num_classes, bias=False)

        # Register masks for pruning
        self.masks = {}
        for name, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):  # Apply to conv & linear layers
                clean_name = name.replace(".", "_")
                mask = torch.ones_like(layer.weight, device='cuda')  # Start with full weight usage
                self.register_buffer(f"mask_{clean_name}", mask)
                self.masks[clean_name] = mask

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Apply masks before forward pass
        for name, layer in self.named_modules():
            cleaned_name = name.replace(".", "_")
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.weight.data *= self.masks[cleaned_name]  # Ensure pruned weights stay zero

        # Two-Path Stem
        local_feat = self.local_conv(x)
        global_feat = self.global_conv(x)
        x = torch.cat([local_feat, global_feat], dim=1)  # Merge both feature maps
        x = self.merge_conv(x)  # Reduce back to 64 channels
        x = F.relu(self.bn1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        # x = F.relu(self.bn3(self.reduced_layer3(x)))
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def kill_small_weights(self, writer=None, global_step=None):
        """Zero out weights below the threshold and prevent updates to them."""

        with torch.no_grad():
            for name, layer in self.named_modules():
                killed_weights = {}
                total_weights = {}
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    name = name.replace(".", "_")
                    if name not in self.masks:
                        continue
                    mask = self.masks[name]  # Get the mask
                    small_weights = layer.weight.abs() < self.kill_threshold  # Find small weights

                    killed_count = small_weights.sum().item()
                    total_count = layer.weight.numel()

                    mask[small_weights] = 0  # Zero out weights in mask
                    layer.weight[small_weights] = 0  # Zero out weights in the layer

                    killed_weights[name] = killed_count
                    total_weights[name] = total_count
                    # Log sparsity to TensorBoard if writer is available
                    if writer is not None and global_step is not None:
                        writer.add_scalar(f"Sparsity/{name}_percent", (killed_count / total_count) * 100, global_step)

        return killed_weights  # Return for logging if needed
