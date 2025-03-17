**DON Concentration Prediction**

This repository contains code for predicting vomitoxin (DON) concentration in corn samples using hyperspectral imaging data. The project includes data preprocessing, neural network model training in PyTorch and evaluation of model performance.

Repository Structure

DON_Concentration_Prediction/

1.app.py                Flask application for model deployment

2. model_training.ipynb  Jupyter Notebook with data preprocessing, training and evaluation

3. README.md             This file

4. Report.pdf

Setup Instructions

Clone the Repository:
```
git clone https://github.com/sreeja-01p/DON_Concentration_Prediction.git
cd DON_Concentration_Prediction
```

Install dependencies which include PyTorch, scikit-learn, pandas, matplotlib, seaborn and flask

Run the Training Notebook:

Open DON_concentration_prediction.ipynb in Colab and follow the instructions to preprocess data, train the model and evaluate performance.
The notebook saves the trained model (model.pth) and the scaler (scaler.pkl).

Run the Flask App:
```python app.py```

The current best model predicts DON concentration with MAE 3815.786, RMSE: 10459.727 and R² score: 0.609
