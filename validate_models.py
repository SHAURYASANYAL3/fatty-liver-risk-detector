import torch
from torchvision import models
import torch.nn as nn
import pickle
import os

print("--- Validating Restored Assets ---")

# 1. Validate CNN Model
cnn_path = "models/train_cnn_robust.pth"
if os.path.exists(cnn_path):
    try:
        model = models.mobilenet_v2(weights=None)
        # Using the same classes count from before (2 classes: Fatty, Normal)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.last_channel, 2)
        )
        model.load_state_dict(torch.load(cnn_path, weights_only=True))
        print(f"[OK] CNN Model ({cnn_path}) loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load CNN Model: {e}")
else:
    print(f"[ERROR] CNN Model not found at {cnn_path}")

# 2. Validate XGBoost Model
xgb_path = "models/masld_xgb_model.pkl"
if os.path.exists(xgb_path):
    try:
        with open(xgb_path, 'rb') as f:
            xgb_model = pickle.load(f)
        print(f"[OK] XGBoost Model ({xgb_path}) loaded successfully!")
        print(f"     Model type: {type(xgb_model)}")
    except Exception as e:
        print(f"[ERROR] Failed to load XGBoost Model: {e}")
else:
    print(f"[ERROR] XGBoost Model not found at {xgb_path}")

print("--- Validation Complete ---")
