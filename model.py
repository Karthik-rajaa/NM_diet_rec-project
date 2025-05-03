import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Sample synthetic dataset
data = pd.DataFrame({
    'Age': [18, 25, 30, 35, 40, 22, 28, 45, 50, 60],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Height': [170, 160, 175, 165, 180, 158, 172, 168, 177, 162],
    'Weight': [65, 55, 80, 60, 85, 52, 75, 58, 90, 56],
    'Activity_Level': ['Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low'],
    'Goal': ['Fat Loss', 'Maintain', 'Muscle Gain', 'Fat Loss', 'Maintain', 'Muscle Gain', 'Fat Loss', 'Maintain', 'Muscle Gain', 'Fat Loss'],
    'Calories': [1800, 2000, 2500, 1750, 2100, 2600, 1700, 2200, 2700, 1650]
})

# Encode categorical columns
label_encoders = {}
for column in ['Gender', 'Activity_Level', 'Goal']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target
X = data.drop('Calories', axis=1)
y = data['Calories']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "calorie_predictor_model.joblib")

print("✅ Model trained and saved as calorie_predictor_model.joblib")
