Breast Cancer Classification with Neural Network 🩺

Overview
This project implements a Neural Network (NN) to classify breast tumors as Malignant or Benign using the Breast Cancer Wisconsin (Diagnostic) dataset from sklearn.datasets. The model is built using TensorFlow and Keras, with data preprocessing steps including standardization and train-test splitting. The project includes training the model, evaluating its performance, and making predictions on new data.
Features

Dataset: Breast Cancer Wisconsin dataset (sklearn.datasets.load_breast_cancer)
Model: Neural Network with 3 layers (Flatten, Dense with ReLU, Dense with Sigmoid)
Preprocessing: Standardization using StandardScaler
Evaluation: Accuracy and loss metrics, visualized with training/validation plots
Prediction: Classifies tumors as Malignant (0) or Benign (1)

Dependencies
To run this project, you need the following Python libraries:

pandas
numpy
matplotlib
scikit-learn
tensorflow

Install them using pip:
pip install pandas numpy matplotlib scikit-learn tensorflow

Project Structure

Breast_Cancer_Classification_with_NN.ipynb: Jupyter Notebook containing the entire project code.
README.md: This file, providing project documentation.

How to Run

Clone the Repository:
git clone https://github.com/<your-username>/breast-cancer-classification-nn.git
cd breast-cancer-classification-nn


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

If you don’t have a requirements.txt, install the dependencies manually as listed above.

Open the Jupyter Notebook:
jupyter notebook Breast_Cancer_Classification_with_NN.ipynb


Run the Notebook:

Execute the cells in the notebook sequentially.
The notebook loads the dataset, preprocesses the data, trains the model, evaluates it, and makes predictions.



Code Breakdown
Data Loading and Preprocessing

Loads the Breast Cancer dataset using sklearn.datasets.load_breast_cancer.
Converts the data into a pandas DataFrame and adds the target labels.
Splits the data into training (80%) and testing (20%) sets.
Standardizes the features using StandardScaler.

Neural Network Model

A Sequential model with:
Flatten layer to convert input into a 1D array.
Dense layer with 20 neurons and ReLU activation.
Dense output layer with 2 neurons and Sigmoid activation (for binary classification).


Compiled with the adam optimizer and sparse_categorical_crossentropy loss.

Training

Trains the model for 10 epochs with a validation split of 0.1.
Plots training and validation accuracy/loss over epochs.

Evaluation

Evaluates the model on the test set and prints the accuracy.

Prediction

Makes predictions on new data by standardizing the input and using the trained model.
Converts predictions to labels (0 for Malignant, 1 for Benign) using argmax.

Results

Model Accuracy: The model achieves an accuracy of around 95-98% on the test set (varies with random seed).
Plots: Training and validation accuracy/loss plots are generated to visualize model performance.
Sample Prediction:
For the input data provided, the model predicts whether the tumor is Malignant or Benign.



Example Prediction
For the input data:
(11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888, 0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769, 12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563)

The model predicts:

Benign (if label is 1) or Malignant (if label is 0), based on the output of argmax.

Future Improvements

Add cross-validation for more robust evaluation.
Experiment with different model architectures (e.g., more layers, dropout).
Include hyperparameter tuning (e.g., learning rate, number of epochs).
Add a user interface for easier input and prediction.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
For questions or contributions, feel free to open an issue or contact the repository owner.
