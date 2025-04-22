import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib

# Sample dataset
data = {
    'funding': [1000000, 5000000, 20000000, 1000000, 3000000],
    'team_size': [5, 15, 50, 4, 10],
    'market_size': [200, 500, 1000, 150, 800],
    'success': [0, 1, 1, 0, 1]  # 0 = failure, 1 = success
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df[['funding', 'team_size', 'market_size']]
y = df['success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'startup_model.pkl')

print("Model trained and saved successfully!")
