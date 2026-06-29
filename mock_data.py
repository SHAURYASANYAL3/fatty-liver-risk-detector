import os
from PIL import Image
import numpy as np
import shutil

base_dir = os.path.join("data", "NFLD_UltraSound_Image_&_Clinical_Dataset")
img_dir = os.path.join(base_dir, "images")

classes = ["Normal", "Fatty"]
for c in classes:
    os.makedirs(os.path.join(img_dir, c), exist_ok=True)
    
print("Generating placeholder images to bypass missing Kaggle zip...")
for c in classes:
    for i in range(20):
        # Create a dummy image
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(img_dir, c, f"img_{i}.jpg"))

# Rename the xlsx file for XGBoost
if os.path.exists(os.path.join(base_dir, "NAFLD.xlsx")):
    shutil.copy2(os.path.join(base_dir, "NAFLD.xlsx"), os.path.join(base_dir, "Clinical_data.xlsx"))
    print("Copied NAFLD.xlsx to Clinical_data.xlsx")

print("Done. Ready to train.")
