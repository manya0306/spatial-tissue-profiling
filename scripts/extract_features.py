import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATCH_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "AD_1_LS"
    / "patches"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "AD_1_LS"
OUTPUT_PATH = OUTPUT_DIR / "resnet18_features.npy"


# Device
device = torch.device("cpu")


# Load pretrained ResNet18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Remove classification layer
model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()

preprocess = weights.transforms()


# Extract features
features = []
barcodes = []

patch_paths = sorted(PATCH_DIR.glob("*.png"))

with torch.no_grad():

    for i, patch_path in enumerate(patch_paths):

        image = Image.open(patch_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        feature = model(image_tensor)
        feature = feature.squeeze(0).numpy()

        features.append(feature)
        barcodes.append(patch_path.stem)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(patch_paths)} patches")


features = np.stack(features)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT_PATH, features)

pd.DataFrame({
    "barcode": barcodes
}).to_csv(
    OUTPUT_DIR / "feature_barcodes.csv",
    index=False,
)

print(f"\nFeature matrix shape: {features.shape}")
print(f"Saved features to: {OUTPUT_PATH}")