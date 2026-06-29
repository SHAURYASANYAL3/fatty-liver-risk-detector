import kagglehub
import shutil
import os

print("Downloading the dataset via kagglehub...")
try:
    path = kagglehub.dataset_download("utkarshx27/non-alcohol-fatty-liver-disease")
    print(f"Downloaded to: {path}")
    
    # The expected data layout was data/NFLD_UltraSound_Image_&_Clinical_Dataset
    # Let's see what's in the downloaded path and copy it to data/NFLD_UltraSound_Image_&_Clinical_Dataset
    
    target_dir = os.path.join("data", "NFLD_UltraSound_Image_&_Clinical_Dataset")
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    shutil.copytree(path, target_dir)
    print(f"Data successfully copied to {target_dir}")
    
    # Also download the other one just in case
    path2 = kagglehub.dataset_download("shaaz74/fatty-liver-progression-dataset-with-biomarkers")
    target_dir2 = os.path.join("data", "fatty-liver-progression-dataset-with-biomarkers")
    if os.path.exists(target_dir2):
        shutil.rmtree(target_dir2)
    shutil.copytree(path2, target_dir2)
    
except Exception as e:
    print(f"Error: {e}")
