# **\<ins\> Breast Cancer Classification with Logistic Regression 🩺\</ins\>**

## Overview

This project implements a Logistic Regression model to classify breast tumors as Malignant or Benign using the Breast Cancer Wisconsin (Diagnostic) dataset from `sklearn.datasets`. The project includes data preprocessing, model training, evaluation, and prediction on new data.

## Features

  * **Dataset:** Breast Cancer Wisconsin dataset (`sklearn.datasets.load_breast_cancer`)
  * **Model:** Logistic Regression
  * **Evaluation:** Accuracy metrics for training and testing data
  * **Prediction:** Classifies tumors as Malignant (0) or Benign (1)

## Dependencies

To run this project, you need the following Python libraries:

  * `pandas`
  * `numpy`
  * `scikit-learn`

Install them using pip:

```bash
pip install pandas numpy scikit-learn
```

## Project Structure

```
Breast_Cancer_Classification.ipynb: Jupyter Notebook containing the entire project code.
README.md: This file, providing project documentation.
```

## How to Run

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/<your-username>/breast-cancer-classification-logistic.git
    cd breast-cancer-classification-logistic
    ```

2.  **Set Up a Virtual Environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    If you don’t have a `requirements.txt`, install the dependencies manually as listed above.

4.  **Open the Jupyter Notebook:**

    ```bash
    jupyter notebook Breast_Cancer_Classification.ipynb
    ```

5.  **Run the Notebook:**
    Execute the cells in the notebook sequentially.
    The notebook loads the dataset, trains the model, evaluates it, and makes predictions.

## Code Breakdown

### Data Loading and Preprocessing

  * Loads the Breast Cancer dataset using `sklearn.datasets.load_breast_cancer`.
  * Converts the data into a pandas DataFrame and adds the target labels.
  * Splits the data into training (80%) and testing (20%) sets.

### Model Training

  * Uses `LogisticRegression` from `sklearn.linear_model` to train the model on the training data.

### Evaluation

  * Evaluates the model on both training and testing sets.
  * Computes accuracy scores for both sets.

### Prediction

  * Makes predictions on new data by reshaping the input and using the trained model.
  * Outputs whether the tumor is Malignant or Benign based on the prediction.

## Results

  * **Training Accuracy:** Achieves an accuracy of around 95% on the training set.
  * **Testing Accuracy:** Achieves an accuracy of around 92% on the test set (varies with random seed).
  * **Sample Prediction:**
    For the input data provided, the model predicts whether the tumor is Malignant or Benign.

### Example Prediction

For the input data:

```
(13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)
```

The model predicts:

**Benign** (if label is 1) or **Malignant** (if label is 0).

## Future Improvements

  * Add data preprocessing steps like standardization to improve model performance.
  * Experiment with other algorithms (e.g., SVM, Random Forest).
  * Include cross-validation for more robust evaluation.
  * Add a user interface for easier input and prediction.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For questions or contributions, feel free to open an issue or contact the repository owner.
