import pandas as pd
import numpy as np
import joblib

def load_model_and_preprocessing():
    """Load the trained model and preprocessing objects."""
    model = joblib.load('/Users/maharshipatel/Downloads/codsfot/sales/sales_model.pkl')
    preprocessing = joblib.load('/Users/maharshipatel/Downloads/codsfot/sales/preprocessing.pkl')
    return model, preprocessing

def preprocess_input(ad_data, preprocessing):
    """Preprocess input data for prediction."""
    df = pd.DataFrame([ad_data])
    
    # Extract preprocessing objects
    scaler = preprocessing['scaler']
    feature_columns = preprocessing['feature_columns']
    
    # Ensure all required columns exist
    required_cols = ['TV', 'Radio', 'Newspaper']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Create additional features
    df['Total_Ad_Spend'] = df['TV'] + df['Radio'] + df['Newspaper']
    df['TV_Radio_Ratio'] = df['TV'] / (df['Radio'] + 1)
    df['TV_Newspaper_Ratio'] = df['TV'] / (df['Newspaper'] + 1)
    df['Radio_Newspaper_Ratio'] = df['Radio'] / (df['Newspaper'] + 1)
    df['TV_Radio_Interaction'] = df['TV'] * df['Radio']
    df['TV_Newspaper_Interaction'] = df['TV'] * df['Newspaper']
    df['Radio_Newspaper_Interaction'] = df['Radio'] * df['Newspaper']
    
    # Select and order features
    # Add missing columns with default values
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_columns]
    
    # Scale features
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_columns)
    
    return df_scaled

def predict_sales(ad_data):
    """Predict sales for given advertising data."""
    # Load model and preprocessing
    model, preprocessing = load_model_and_preprocessing()
    
    # Preprocess input
    X = preprocess_input(ad_data, preprocessing)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction

def main():
    """Example usage of the prediction function."""
    print("=" * 60)
    print("SALES PREDICTION")
    print("=" * 60)
    
    # Example advertising data
    example_ad = {
        'TV': 200,      # TV advertising spend in thousands
        'Radio': 30,    # Radio advertising spend in thousands
        'Newspaper': 40 # Newspaper advertising spend in thousands
    }
    
    print("\nExample Advertising Budget:")
    for key, value in example_ad.items():
        print(f"  {key}: ${value:.1f}k")
    
    print(f"\nTotal Advertising Budget: ${sum(example_ad.values()):.1f}k")
    
    # Make prediction
    prediction = predict_sales(example_ad)
    
    print("\nPrediction Result:")
    print(f"  Predicted Sales: {prediction:.2f}k units")
    print(f"  Predicted Sales: ${prediction * 1000:.2f}")
    
    # Calculate ROI scenarios
    print("\n" + "=" * 60)
    print("OPTIMIZATION SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        {'TV': 250, 'Radio': 40, 'Newspaper': 50, 'name': 'High Budget'},
        {'TV': 150, 'Radio': 25, 'Newspaper': 30, 'name': 'Medium Budget'},
        {'TV': 100, 'Radio': 20, 'Newspaper': 20, 'name': 'Low Budget'},
        {'TV': 300, 'Radio': 10, 'Newspaper': 10, 'name': 'TV Focused'},
        {'TV': 50, 'Radio': 50, 'Newspaper': 50, 'name': 'Balanced'},
    ]
    
    print("\nBudget Allocation Scenarios:")
    print(f"{'Scenario':<20} {'TV':<10} {'Radio':<10} {'Newspaper':<12} {'Total':<10} {'Predicted Sales':<15}")
    print("-" * 85)
    
    for scenario in scenarios:
        name = scenario.pop('name')
        total = sum(scenario.values())
        prediction = predict_sales(scenario)
        print(f"{name:<20} {scenario['TV']:<10} {scenario['Radio']:<10} {scenario['Newspaper']:<12} {total:<10} {prediction:<15.2f}")
    
    print("\n" + "=" * 60)
    print("To make predictions for your own data:")
    print("1. Import the predict_sales function")
    print("2. Pass a dictionary with advertising spend information")
    print("3. Required fields: TV, Radio, Newspaper (all in thousands)")
    print("=" * 60)

if __name__ == "__main__":
    main()
