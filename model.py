import torch
import torch.nn as nn
import torchvision.models as models


class CausalTSM(nn.Module):
    """
    Causal Temporal Shift Module (TSM).
    Shifts 1/8th of feature map channels along the time dimension strictly 
    from the past frame to the present frame (t-1 -> t).
    
    Zero added parameters, zero added FLOPs.
    """
    def __init__(self, n_segment=16, fold_div=8):
        super(CausalTSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        # Input shape: (B * T, C, H, W)
        bt, c, h, w = x.size()
        b = bt // self.n_segment
        t = self.n_segment

        # Reshape to (B, T, C, H, W) to perform temporal shift
        x = x.view(b, t, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)

        # 1. Causal Shift: Copy channels from t-1 into t
        out[:, 1:, :fold, :, :] = x[:, :-1, :fold, :, :]
        # Keep initial frame t=0 padded with its own channels
        out[:, 0, :fold, :, :] = x[:, 0, :fold, :, :]

        # 2. Unshifted remaining channels (7/8th of total channels)
        out[:, :, fold:, :, :] = x[:, :, fold:, :, :]

        # Flatten back to (B * T, C, H, W) for 2D MobileNet convolutions
        return out.view(bt, c, h, w)


class PhotosensitiveMobileNetTSM(nn.Module):
    """
    MobileNetV3-Small augmented with Causal TSM for Unified Hazard Classification:
    - Output: 1 single logit (0 = Safe / Clean, 1 = Visual Hazard)
    """
    def __init__(self, num_classes=1, num_segments=16, pretrained=True):
        super(PhotosensitiveMobileNetTSM, self).__init__()
        self.num_segments = num_segments

        # Load MobileNetV3-Small backbone with ImageNet pretrained weights
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Inject Causal TSM into every InvertedResidual block in MobileNetV3
        self.features = nn.ModuleList()
        for layer in backbone.features:
            if isinstance(layer, models.mobilenetv3.InvertedResidual):
                self.features.append(nn.Sequential(
                    CausalTSM(n_segment=num_segments, fold_div=8),
                    layer
                ))
            else:
                self.features.append(layer)

        # Global Spatial Average Pooling
        self.avgpool = backbone.avgpool

        # Classification Head (1 Output for Unified Hazard Detection)
        in_features = backbone.classifier[3].in_features
        self.classifier = nn.Sequential(
            backbone.classifier[0],  # Linear
            backbone.classifier[1],  # Hardswish
            backbone.classifier[2],  # Dropout
            nn.Linear(in_features, num_classes)  # Output shape: (B, 1)
        )

    def forward(self, x):
        """
        Forward pass for unified hazard classification.
        
        Args:
            x (torch.Tensor): Preprocessed input tensor of shape (B, 16, 3, 224, 224)
            
        Returns:
            logits (torch.Tensor): Raw classification logit of shape (B, 1)
        """
        # x shape: (B, T, C, H, W) -> e.g. (32, 16, 3, 224, 224)
        b, t, c, h, w = x.size()

        # 1. Fold Batch and Time dimensions for standard 2D convolutions
        x = x.view(b * t, c, h, w)  # Shape: (B * 16, 3, 224, 224)

        # 2. Extract spatiotemporal features through MobileNetV3 + TSM
        for layer in self.features:
            x = layer(x)

        # 3. Global Spatial Average Pooling -> Shape: (B * 16, C_out, 1, 1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)     # Shape: (B * 16, C_out)

        # 4. Temporal Mean Pooling across the 16 frame time-steps
        x = x.view(b, t, -1)        # Shape: (B, 16, C_out)
        video_features = x.mean(dim=1)  # Shape: (B, C_out)

        # 5. Classification Head
        logits = self.classifier(video_features)  # Shape: (B, 1)
        return logits


def build_model(num_classes=1, num_segments=16, pretrained=True):
    """
    Helper function to instantiate the unified single-output model.
    """
    return PhotosensitiveMobileNetTSM(
        num_classes=num_classes,
        num_segments=num_segments,
        pretrained=pretrained
    )


if __name__ == "__main__":
    print("Initializing Unified Photosensitive MobileNetV3 + Causal TSM Model...")
    model = build_model(num_classes=1, num_segments=16, pretrained=True)
    model.eval()

    # Create dummy input batch: Batch=2, Time-steps=16, Channels=3, Height=224, Width=224
    dummy_input = torch.randn(2, 16, 3, 224, 224)

    with torch.no_grad():
        logits = model(dummy_input)
        probability = torch.sigmoid(logits)

    print("\n--- Sanity Check Results ---")
    print(f"Input Shape:         {dummy_input.shape}")
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Sample Hazard Probability (p_hazard):\n{probability}")
    print("\nModel architecture verified successfully!")
