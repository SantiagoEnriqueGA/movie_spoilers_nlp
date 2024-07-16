import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

# Load the data splits
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')

# Models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, verbose=1, n_jobs=-1),
    'SGD Classifier': SGDClassifier(random_state=42, verbose=1, n_jobs=-1),
    'Random Forest': RandomForestClassifier(random_state=42, verbose=1, n_jobs=-1),
    'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1), 
    'XGBoost': XGBClassifier(random_state=42, verbosity=1, n_jobs=-1),
    'LightGBM': LGBMClassifier(random_state=42, verbose=1, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, verbose=1),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Linear SVC': LinearSVC(random_state=42, verbose=1),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f'Training {model_name}...')
    
    # Convert sparse matrices to dense arrays for Naive Bayes
    if model_name == 'Naive Bayes':
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        model.fit(X_train_dense, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Evaluate the model
    if model_name == 'Naive Bayes':
        y_pred = model.predict(X_test_dense)
    else:
        y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    print(f'Classification Report for {model_name}:')
    print(report)
    
    # Save the trained model
    joblib.dump(model, f'models/v2/base/{model_name.replace(" ", "_").lower()}_model.pkl')
    print(f'Saved {model_name} model.\n')