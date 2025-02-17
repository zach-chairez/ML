# Machine Learning Models Overview

## 1. Logistic Regression
### Description:
Logistic Regression is a statistical method used for binary classification problems. It predicts the probability that a given input belongs to a certain class using the logistic (sigmoid) function.

### Utility:
- Used for classification tasks where the target variable is binary (e.g., spam vs. non-spam, fraud detection).
- Provides interpretable feature importance through model coefficients.

### Computational Complexity:
- **Training:** $O(n \cdot m)$, where $n$ is the number of samples and $m$ is the number of features.
- **Prediction:** $O(m)$, as it requires computing a weighted sum of the features.

### Python Implementation:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

---

## 2. Time Series Analysis
### Description:
Time Series Analysis focuses on analyzing temporal data points to identify trends, seasonal patterns, and forecasting future values.

### Utility:
- Used in financial forecasting, sales prediction, and anomaly detection in sensor data.
- Helps understand trends and cyclical patterns in data.

### Computational Complexity:
- **Simple Models (AR, MA, ARMA):** $O(n)$
- **More Complex Models (ARIMA, SARIMA, LSTM):** $O(n^2)$ or worse

### Python Implementation:
```python
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Simulate time series data
np.random.seed(42)
time_series = np.cumsum(np.random.randn(100))

# Fit ARIMA model
model = ARIMA(time_series, order=(1, 1, 1))  # ARIMA(p,d,q)
model_fit = model.fit()

# Forecast the next 10 steps
forecast = model_fit.forecast(steps=10)
```

---

## 3. Decision Trees
### Description:
Decision trees recursively split data based on feature values to create a hierarchy of decisions.

### Utility:
- Used in classification and regression tasks.
- Easy to interpret and visualize.

### Computational Complexity:
- **Training:** $O(nm \log n)$
- **Prediction:** $O(\log n)$

### Python Implementation:
```python
from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

---

## 4. Gradient-Boosted Trees
### Description:
Gradient Boosting builds an ensemble of weak decision trees, iteratively improving predictions by correcting residual errors.

### Utility:
- Used for structured data problems like fraud detection and risk assessment.
- More powerful than standalone decision trees.

### Computational Complexity:
- **Training:** $O(nmT)$, where $T$ is the number of trees.
- **Prediction:** $O(T)$.

### Python Implementation:
```python
from sklearn.ensemble import GradientBoostingClassifier

# Train model
model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

---

## 5. Neural Networks
### Description:
Neural networks use layers of interconnected neurons to learn complex patterns in data.

### Utility:
- Used in image recognition, NLP, and deep learning applications.
- Captures non-linear relationships effectively.

### Computational Complexity:
- **Training:** $O(nmL)$, where $L$ is the number of layers.
- **Prediction:** $O(mL)$.

### Python Implementation (Using TensorFlow):
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define a simple neural network
model = Sequential([
    Dense(50, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
```

---

## Summary Table:
| Model | Type | Complexity (Training) | Best Use Case |
|--------|-------|---------------------|--------------|
| Logistic Regression | Classification | $O(nm)$ | Binary classification |
| Time Series (ARIMA) | Forecasting | $O(n^2)$ | Temporal data trends |
| Decision Trees | Classification | $O(nm \log n)$ | Rule-based models |
| Gradient-Boosted Trees | Classification | $O(nmT)$ | High accuracy on structured data |
| Neural Networks | Deep Learning | $O(nmL)$ | Complex, high-dimensional data |

This markdown file serves as a guide to common machine learning models, their applications, and their implementations in Python.
