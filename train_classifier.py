import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from monai.data import ImageDataset
from src.models import TripletResNet, TripletResNetClassifier
from src.utils import get_transforms
from sklearn.metrics import accuracy_score

# Configuration
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Update this path after running the metric learning step
PRETRAINED_PATH = "results/metric_learning/models/trunk_best.pth" 

def train():
    # 1. Load Data
    train_df = pd.read_csv('data/train.csv')
    valid_df = pd.read_csv('data/valid.csv')
    train_transforms, val_transforms = get_transforms()

    train_ds = ImageDataset(train_df['scan'].tolist(), train_df['label'].tolist(), transform=train_transforms)
    valid_ds = ImageDataset(valid_df['scan'].tolist(), valid_df['label'].tolist(), transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Load Pretrained Backbone
    backbone = TripletResNet(device=DEVICE)

    try:
        # Load weights, handling potential strict/non-strict keys from PML
        if torch.cuda.is_available():
            checkpoint = torch.load(PRETRAINED_PATH)
        else:
            checkpoint = torch.load(PRETRAINED_PATH, map_location=torch.device('cpu'))
            
        state_dict = checkpoint.get('trunk_state_dict', checkpoint)
        backbone.load_state_dict(state_dict, strict=False)
        print("Pretrained backbone loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: Pretrained model not found at {PRETRAINED_PATH}. Initializing random weights.")

    # Freeze Backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # 3. Create Classifier Wrapper
    model = TripletResNetClassifier(backbone, device=DEVICE)
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE).float()
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE).float()
                outputs = model(inputs).squeeze()
                preds = torch.round(torch.sigmoid(outputs))
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.numpy())
        
        acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_classifier.pth")
            print(f"Saved new best model with accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()