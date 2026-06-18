# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score, roc_auc_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
from sklearn.metrics import confusion_matrix

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_experiment")

api = HfApi()

DATASET_PATH = "hf://datasets/vishaldixit75/tourismData/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

Xtrain_path = "hf://datasets/vishaldixit75/tourismData/Xtrain.csv"
Xtest_path = "hf://datasets/vishaldixit75/tourismData/Xtest.csv"
ytrain_path = "hf://datasets/vishaldixit75/tourismData/ytrain.csv"
ytest_path = "hf://datasets/vishaldixit75/tourismData/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


print(f"Training set shape: {Xtrain.shape}")
print(f"Test set shape: {Xtest.shape}")

# Identify numeric features (all features after encoding)
numeric_features = Xtrain.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features= Xtrain.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="drop"
)

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75],
    'xgbclassifier__max_depth': [2, 3],
    'xgbclassifier__colsample_bytree': [0.4, 0.5],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5],
    'xgbclassifier__learning_rate': [0.01, 0.05],
    'xgbclassifier__reg_lambda': [0.4, 0.5],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.5

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Calculate additional metrics for logging
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)
    train_precision = precision_score(ytrain, y_pred_train, zero_division=0)
    test_precision = precision_score(ytest, y_pred_test, zero_division=0)
    train_recall = recall_score(ytrain, y_pred_train, zero_division=0)
    test_recall = recall_score(ytest, y_pred_test, zero_division=0)
    train_f1 = f1_score(ytrain, y_pred_train, zero_division=0)
    test_f1 = f1_score(ytest, y_pred_test, zero_division=0)
    train_roc_auc = roc_auc_score(ytrain, y_pred_train_proba)
    test_roc_auc = roc_auc_score(ytest, y_pred_test_proba)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "train_roc_auc": train_roc_auc,
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score'],
        "test_roc_auc": test_roc_auc
    })

    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f} | Test Precision: {test_precision:.4f}")
    print(f"Train Recall: {train_recall:.4f} | Test Recall: {test_recall:.4f}")
    print(f"Train F1-Score: {train_f1:.4f} | Test F1-Score: {test_f1:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")
    print("="*50)

    print("\nTest Set Classification Report:")
    print(classification_report(ytest, y_pred_test, target_names=['No Purchase', 'Purchase']))

    print("\nTest Set Confusion Matrix:")
    print(confusion_matrix(ytest, y_pred_test))

    # Save the model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved locally as: {model_path}")

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model logged to MLflow")

    # Upload to Hugging Face
    repo_id = "vishaldixit75/tourismData"
    repo_type = "model"

    # Step 1: Check if the repository exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"\nRepository '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"\nRepository '{repo_id}' not found. Creating new repository...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repository '{repo_id}' created.")

    # Upload model to Hugging Face
    api.upload_file(
        path_or_fileobj="best_tourism_model_v1.joblib",
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face: {repo_id}")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
