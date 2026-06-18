# for data manipulation
import pandas as pd
import numpy as np
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
import os

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/vishaldixit75/tourismData/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# Drop the unnamed index column if it exists
if 'Unnamed: 0' in df.columns or df.columns[0] == '':
    df = df.iloc[:, 1:]

# Drop CustomerID as it's a unique identifier (not useful for modeling)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Handle missing values
print("\nHandling missing values...")

df["Gender"]=np.where(df["Gender"]=="Fe Male","Female",df["Gender"])
df["MaritalStatus"]=np.where(df["MaritalStatus"]=="Single","Unmarried",df["MaritalStatus"])

target_col = "ProdTaken"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

#creating dummy variables
X = pd.get_dummies(
    X,
    columns=X.select_dtypes(include=["object", "category"]).columns.tolist(),
    drop_first=True,
)

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Xtrain shape after split: {Xtrain.shape}")
print(f"ytrain shape after split: {ytrain.shape}")
print(f"Xtest shape after split: {Xtest.shape}")
print(f"ytest shape after split: {ytest.shape}")


Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

df["ProdTaken"].value_counts()  #it is not the Imbalanced Dataset

files_to_upload = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv"
]

for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,  # just the filename
        repo_id="vishaldixit75/tourismData",
        repo_type="dataset",
    )
    print(f"Successfully uploaded {file_path} to Hugging Face.")
