import os
import torch
import pandas as pd
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from monai.data import ImageDataset
from src.models import TripletResNet
from src.utils import get_transforms

# Configuration
BATCH_SIZE = 32
EPOCHS = 400
MARGIN = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results/metric_learning"

def main():
    # 1. Load Data
    # Ensure you have 'scan' and 'label' columns in your CSVs
    if not os.path.exists('data/train.csv'):
        print("Error: data/train.csv not found. Please prepare your dataset first.")
        return

    train_df = pd.read_csv('data/train.csv')
    valid_df = pd.read_csv('data/valid.csv')

    train_transforms, val_transforms = get_transforms()

    train_ds = ImageDataset(
        image_files=train_df['scan'].tolist(), 
        labels=train_df['label'].tolist(), 
        transform=train_transforms
    )
    valid_ds = ImageDataset(
        image_files=valid_df['scan'].tolist(), 
        labels=valid_df['label'].tolist(), 
        transform=val_transforms
    )

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(valid_ds)}")

    # 2. Setup Model & Metric Learning Components
    model = TripletResNet(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    
    loss_func = losses.TripletMarginLoss(margin=MARGIN, swap=True, smooth_loss=True)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    
    # Sampler ensures we get m=4 instances per class in every batch
    sampler = samplers.MPerClassSampler(train_ds.labels, m=4, length_before_new_iter=len(train_ds))

    # 3. Trainer Setup
    record_keeper, _, _ = logging_presets.get_record_keeper(f"{OUTPUT_DIR}/logs", f"{OUTPUT_DIR}/tensorboard")
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"train": train_ds, "val": valid_ds}

    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        dataloader_num_workers=2,
        use_trunk_output=False,
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, f"{OUTPUT_DIR}/models", test_interval=20
    )

    trainer = trainers.MetricLossOnly(
        models={"trunk": model},
        optimizers={"trunk_optimizer": optimizer},
        batch_size=BATCH_SIZE,
        loss_funcs={"metric_loss": loss_func},
        mining_funcs={"tuple_miner": miner},
        dataset=train_ds,
        sampler=sampler,
        dataloader_num_workers=2,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )

    # 4. Train
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.train(num_epochs=EPOCHS)

if __name__ == "__main__":
    main()