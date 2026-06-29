import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle

def main():
    data_path = os.path.join("data", "NFLD_UltraSound_Image_&_Clinical_Dataset", "Clinical_data.xlsx")
    if not os.path.exists(data_path):
        print(f"Error: Clinical data not found at {data_path}")
        print("Please download the Kaggle dataset into the data/ folder.")
        return

    # Load data
    df = pd.read_excel(data_path)
    
    # Preprocessing (Assume simple cleaning based on context)
    # We drop ID or subjective columns, and use metabolic markers
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    # Assume target column is 'Diagnosis' or similar; handle robustly
    target_col = [c for c in df.columns if 'diagnos' in c.lower() or 'class' in c.lower()]
    if not target_col:
        target_col = df.columns[-1] # Fallback to last column
    else:
        target_col = target_col[0]

    # Convert targets to binary (0 and 1)
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype('category').cat.codes
    
    # Ensure it starts at 0
    df[target_col] = df[target_col] - df[target_col].min()

    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"XGBoost Clinical Model Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "masld_xgb_model.pkl"), 'wb') as f:
        pickle.dump(model, f)
    
    print("Clinical XGBoost model saved to models/masld_xgb_model.pkl")

if __name__ == "__main__":
    main()
