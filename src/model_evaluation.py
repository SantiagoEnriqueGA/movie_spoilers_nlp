import joblib
import pandas as pd
from sklearn.metrics import classification_report
import os

# Load the datasets
print("Loading datasets...")
X_test_base = joblib.load('data/processed/splits/base/X_test.pkl')
y_test_base = joblib.load('data/processed/splits/base/y_test.pkl')

X_test_smote = joblib.load('data/processed/splits/smote/X_test.pkl')
y_test_smote = joblib.load('data/processed/splits/smote/y_test.pkl')

X_test_smote_pca = joblib.load('data/processed/splits/smote_pca/X_test.pkl')
y_test_smote_pca = joblib.load('data/processed/splits/smote_pca/y_test.pkl')

# Define model paths
model_paths = {
    'base': 'models/base/',
    'smote': 'models/smote/',
    'smote_pca': 'models/smote_pca/'
}

# Define a function to load and evaluate models
def evaluate_model(model_path, X_test, y_test):
    print(f"Evaluating model: {model_path}")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

# Define a function to flatten reports into a DataFrame
def flatten_report(report, model_name):
    flatten = {}
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flatten[f"{key}_{sub_key}"] = sub_value
        else:
            flatten[key] = value
    flatten['model'] = model_name
    return flatten

# Load and evaluate models
reports = []
for key, path in model_paths.items():
    print(f"Processing models in: {path}")
    for model_file in os.listdir(path):
        if model_file.endswith('_model.pkl'):
            model_name = f"{key}_{model_file.split('_model.pkl')[0]}"
            print(f"Evaluating {model_name}...")
            if key == 'base':
                report = evaluate_model(os.path.join(path, model_file), X_test_base, y_test_base)
            elif key == 'smote':
                report = evaluate_model(os.path.join(path, model_file), X_test_smote, y_test_smote)
            elif key == 'smote_pca':
                report = evaluate_model(os.path.join(path, model_file), X_test_smote_pca, y_test_smote_pca)
            reports.append(flatten_report(report, model_name))

# Convert the list of flattened reports to a DataFrame
print("Converting reports to DataFrame...")
report_df = pd.DataFrame(reports)

# Save the DataFrame to a CSV file
output_file = 'data/processed/evaluation_reports.csv'
report_df.to_csv(output_file, index=False)
print(f"Evaluation reports saved to '{output_file}'")

# Load and print the report file
print("Loading and printing the evaluation report...")
loaded_report_df = pd.read_csv(output_file)
print(loaded_report_df)
