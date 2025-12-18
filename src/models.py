import torch
import torch.nn as nn
from monai.networks.nets import Classifier

class TripletResNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        # The Trunk (MONAI Classifier)
        # Matches your paper's configuration with classes=5000
        self.trunk = Classifier(
            in_shape=(1, 128, 128, 64), 
            classes=5000, 
            channels=(32, 64, 128, 256), 
            num_res_units=3, 
            strides=(2, 2, 3, 3)
        ).to(device)

        # The Embedder (MLP)
        # Output is 512-dimensional embedding
        self.embedder = nn.Sequential(
            nn.Linear(5000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        ).to(device)

    def forward(self, x):
        x = self.trunk(x)
        x = self.embedder(x)
        return x

class TripletResNetClassifier(nn.Module):
    def __init__(self, pretrained_trunk_model, device):
        super().__init__()
        self.feature_extractor = pretrained_trunk_model
        
        # Classification Head
        # Takes the 512 embedding and outputs a single logit (Binary Classification)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        ).to(device)

    def forward(self, x):
        # We assume the trunk is frozen during training
        embeddings = self.feature_extractor(x)
        return self.classifier(embeddings)