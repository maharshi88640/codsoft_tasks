import pandas as pd
import numpy as np
import joblib

def load_model_and_preprocessing():
    """Load the trained model and preprocessing objects."""
    model = joblib.load('/Users/maharshipatel/Downloads/codsfot/titanic/titanic_model.pkl')
    preprocessing = joblib.load('/Users/maharshipatel/Downloads/codsfot/titanic/preprocessing.pkl')
    return model, preprocessing

def preprocess_input(passenger_data, preprocessing):
    """Preprocess input data for prediction."""
    df = pd.DataFrame([passenger_data])
    
    # Extract preprocessing objects
    scaler = preprocessing['scaler']
    le_sex = preprocessing['le_sex']
    le_embarked = preprocessing['le_embarked']
    feature_columns = preprocessing['feature_columns']
    
    # Ensure all required columns exist
    required_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Encode categorical variables
    df['Sex'] = le_sex.transform(df['Sex'])
    df['Embarked'] = le_embarked.transform(df['Embarked'])
    
    # Create features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Select and order features
    df = df[feature_columns]
    
    # Scale numerical features
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

def predict_survival(passenger_data):
    """Predict survival for a passenger."""
    # Load model and preprocessing
    model, preprocessing = load_model_and_preprocessing()
    
    # Preprocess input
    X = preprocess_input(passenger_data, preprocessing)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return prediction, probability

def main():
    """Example usage of the prediction function."""
    print("=" * 50)
    print("TITANIC SURVIVAL PREDICTION")
    print("=" * 50)
    
    # Example passenger data
    example_passenger = {
        'Pclass': 3,        # Passenger class (1, 2, or 3)
        'Sex': 'male',     # 'male' or 'female'
        'Age': 22,         # Age in years
        'SibSp': 1,        # Number of siblings/spouses aboard
        'Parch': 0,        # Number of parents/children aboard
        'Fare': 7.25,      # Ticket fare
        'Embarked': 'S'    # Port of embarkation ('S', 'C', or 'Q')
    }
    
    print("\nExample Passenger:")
    for key, value in example_passenger.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    prediction, probability = predict_survival(example_passenger)
    
    print("\nPrediction Result:")
    print(f"  Survived: {'Yes' if prediction == 1 else 'No'}")
    print(f"  Survival Probability: {probability:.2%}")
    
    print("\n" + "=" * 50)
    print("To make predictions for your own data:")
    print("1. Import the predict_survival function")
    print("2. Pass a dictionary with passenger information")
    print("3. Required fields: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")
    print("=" * 50)

if __name__ == "__main__":
    main()
