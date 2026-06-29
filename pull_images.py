import kagglehub
import shutil
import os

print("Downloading the image dataset via kagglehub...")
try:
    path = kagglehub.dataset_download("shanecandoit/dataset-of-bmode-fatty-liver-ultrasound-images")
    print(f"Downloaded to: {path}")
    
    target_dir = os.path.join("data", "NFLD_UltraSound_Image_&_Clinical_Dataset")
    
    # Do not clear target_dir completely as we might need the csv files or clinical data
    # Just copy the files over
    for root, dirs, files in os.walk(path):
        for file in files:
            src_path = os.path.join(root, file)
            # Find relative path
            rel_path = os.path.relpath(src_path, path)
            dest_path = os.path.join(target_dir, rel_path)
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
    
    print(f"Image Data successfully copied to {target_dir}")
except Exception as e:
    print(f"Error: {e}")
