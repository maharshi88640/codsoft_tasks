# Titanic Survival Prediction Model

A machine learning model that predicts whether a passenger on the Titanic survived or not based on passenger characteristics.

## Project Overview

This classic beginner ML project uses the Titanic dataset to build a binary classification model. The model predicts survival (1) or non-survival (0) based on features like passenger class, age, sex, fare, and family size.

## Dataset

The dataset contains 891 passengers with the following features:
- **PassengerId**: Unique identifier for each passenger
- **Survived**: Target variable (0 = Did not survive, 1 = Survived)
- **Pclass**: Passenger class (1 = First, 2 = Second, 3 = Third)
- **Name**: Passenger name
- **Sex**: Passenger gender (male/female)
- **Age**: Passenger age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (S = Southampton, C = Cherbourg, Q = Queenstown)

## Model Performance

The best performing model is **Logistic Regression** with:
- **Cross-validation accuracy**: 80.19%
- **Test accuracy**: 79.89%
- **Precision (Did Not Survive)**: 82%
- **Recall (Did Not Survive)**: 85%
- **Precision (Survived)**: 77%
- **Recall (Survived)**: 73%

## Features Used

The model uses the following engineered features:
- Pclass (Passenger class)
- Sex (Encoded)
- Age (Imputed with median)
- SibSp (Siblings/spouses)
- Parch (Parents/children)
- Fare (Ticket fare)
- Embarked (Encoded)
- FamilySize (SibSp + Parch + 1)
- IsAlone (Binary: 1 if FamilySize == 1, else 0)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model from scratch:
```bash
python3 titanic_model.py
```

This will:
- Load and explore the dataset
- Create visualizations
- Preprocess the data
- Train and evaluate models
- Save the best model and preprocessing objects
- Generate performance plots

### Making Predictions

To make predictions on new passenger data:
```bash
python3 predict.py
```

Or use the prediction function in your own code:
```python
from predict import predict_survival

passenger_data = {
    'Pclass': 3,
    'Sex': 'male',
    'Age': 22,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 7.25,
    'Embarked': 'S'
}

prediction, probability = predict_survival(passenger_data)
print(f"Survived: {'Yes' if prediction == 1 else 'No'}")
print(f"Survival Probability: {probability:.2%}")
```

## Files Generated

- **titanic_model.pkl**: Trained model
- **preprocessing.pkl**: Preprocessing objects (scaler, encoders)
- **titanic_visualizations.png**: Data exploration plots
- **confusion_matrix.png**: Model confusion matrix
- **feature_importance.png**: Feature importance plot

## Key Findings

1. **Gender**: Females had a significantly higher survival rate than males
2. **Class**: First-class passengers had higher survival rates than third-class
3. **Age**: Children and young adults had better survival rates
4. **Family Size**: Small families (2-4 members) had better survival rates
5. **Embarkation**: Passengers from Cherbourg had higher survival rates

## Model Architecture

The project compares two models:
1. **Random Forest Classifier**: 100 estimators
2. **Logistic Regression**: Maximum 1000 iterations

Logistic Regression performed slightly better on this dataset.

## Data Preprocessing

- Missing values in `Age` filled with median
- Missing values in `Embarked` filled with mode
- `Cabin` column dropped (687 missing values)
- Categorical variables encoded using LabelEncoder
- Numerical features standardized using StandardScaler
- Engineered features: FamilySize and IsAlone

## Future Improvements

- Try more advanced models (XGBoost, LightGBM)
- Perform hyperparameter tuning
- Use more sophisticated feature engineering
- Implement cross-validation with stratified splits
- Add more detailed error analysis
