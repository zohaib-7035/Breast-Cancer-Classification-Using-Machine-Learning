# Breast Cancer Classification using Logistic Regression

## Overview
This project implements a **Breast Cancer Classification model** using **Logistic Regression** in Python. The dataset used is the **Breast Cancer Wisconsin dataset** from Scikit-learn. The goal is to classify breast tumors as **Malignant (cancerous) or Benign (non-cancerous)** based on given features.

## Dataset
The dataset is loaded using:
```python
from sklearn.datasets import load_breast_cancer
breast_cancer_dataset = load_breast_cancer()
```
### Features:
- 30 numerical features describing tumor characteristics
- **Target Variable:**
  - `0` → Malignant (cancerous)
  - `1` → Benign (non-cancerous)

## Project Structure
- `Breast Cancer Classification.ipynb` → Jupyter Notebook containing the full implementation.
- `README.md` → Documentation for the project.
- `requirements.txt` → List of dependencies required to run the project.


## Implementation Steps
1. **Load and Explore Data**
   - Convert dataset to a Pandas DataFrame.
   - Check dataset properties using `.shape`, `.info()`, and `.describe()`.
2. **Data Preprocessing**
   - Check for missing values.
   - Add `label` column (target values).
3. **Train-Test Split**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
   ```
4. **Train the Model**
   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression()
   model.fit(X_train, Y_train)
   ```
5. **Evaluate the Model**
   - Calculate accuracy using `accuracy_score`.
   ```python
   from sklearn.metrics import accuracy_score
   accuracy_score(Y_test, model.predict(X_test))
   ```
6. **Make Predictions**
   - Take user input and predict if the tumor is **Malignant or Benign**.

## Results
- Training Accuracy: **~95%**
- Testing Accuracy: **~96%**

## Usage
To predict a single instance:
```python
input_data = (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)

prediction = model.predict([input_data])
print('Malignant' if prediction[0] == 0 else 'Benign')
```

## Dependencies
- Python 3
- Pandas
- NumPy
- Scikit-learn

---
Feel free to modify and improve this project! 🚀

