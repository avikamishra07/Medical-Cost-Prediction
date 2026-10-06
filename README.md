# Medical Cost Prediction System

## Project Overview

This project predicts the annual medical cost of a patient using machine learning regression models.

The project includes data preprocessing, exploratory data analysis, correlation and covariance analysis, model training, model comparison, and a Streamlit prediction interface.

## Dataset

The dataset used for this project is a medical insurance dataset containing patient-related information and annual medical cost.

### Target Variable

`annual_medical_cost`

## Machine Learning Models

The following regression models were implemented:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regression
* Random Forest Regression

The models were evaluated using:

* MAE
* MSE
* RMSE
* R² Score

The model with the best R² score was selected as the final model.

## Streamlit Application

The Streamlit application allows the user to enter patient details such as:

* Age
* Income
* Household Size
* Dependents
* BMI
* Visits Last Year
* Risk Score

The application then predicts the annual medical cost using the saved machine learning model.

## Project Files

```text
medical-cost-prediction/
│
├── app.py
├── best_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── model_name.pkl
├── .gitignore
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib

## How to Run

Install the required libraries:

```bash
pip install streamlit numpy pandas scikit-learn joblib matplotlib
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## Output

The application displays the predicted annual medical cost and the machine learning model used for prediction.

## Author

Avika Mishra
