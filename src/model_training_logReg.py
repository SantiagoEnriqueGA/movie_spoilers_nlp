import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the data splits
X_train = joblib.load('data/processed/splits/X_train.pkl')
X_test = joblib.load('data/processed/splits/X_test.pkl')
y_train = joblib.load('data/processed/splits/y_train.pkl')
y_test = joblib.load('data/processed/splits/y_test.pkl')

# Load the TF-IDF vectorizer and scaler
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
scaler = joblib.load('models/scaler.pkl')

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'models/logistic_regression_model.pkl')
