import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "hours": [1, 2, 3, 4, 5],
    "attendance": [50, 60, 70, 80, 90],
    "score": [40, 50, 65, 70, 85]
}

df = pd.DataFrame(data)

# Features and target
X = df[["hours", "attendance"]]
y = df["score"]

# Model
model = LinearRegression()
model.fit(X, y)

# Predict

new_data = pd.DataFrame([[6, 95]], columns=["hours", "attendance"])
prediction = model.predict(new_data)

print("Predicted Score:", prediction[0])
