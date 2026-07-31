import pandas as pd
import numpy as np
import joblib

def load_model_and_preprocessing():
    """Load the trained model and preprocessing objects."""
    model = joblib.load('/Users/maharshipatel/Downloads/codsfot/imdb/movie_rating_model.pkl')
    preprocessing = joblib.load('/Users/maharshipatel/Downloads/codsfot/imdb/preprocessing.pkl')
    return model, preprocessing

def preprocess_input(movie_data, preprocessing):
    """Preprocess input data for prediction."""
    df = pd.DataFrame([movie_data])
    
    # Extract preprocessing objects
    scaler = preprocessing['scaler']
    le_director = preprocessing['le_director']
    top_genres = preprocessing['top_genres']
    feature_columns = preprocessing['feature_columns']
    
    # Ensure all required columns exist
    required_cols = ['Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Handle Year
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Handle Duration - extract minutes
    df['Duration_min'] = df['Duration'].str.extract(r'(\d+)').astype(float)
    df = df.drop(['Duration'], axis=1)
    
    # Handle Votes - convert to numeric
    df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    
    # Create genre features
    for genre in top_genres:
        df[f'Genre_{genre}'] = df['Genre'].apply(lambda x: 1 if genre in str(x) else 0)
    df = df.drop(['Genre'], axis=1)
    
    # Handle Director encoding
    # Check if director is in the encoder's classes
    director = df['Director'].iloc[0]
    if director in le_director.classes_:
        df['Director_encoded'] = le_director.transform([director])[0]
    else:
        df['Director_encoded'] = le_director.transform(['Other'])[0]
    df = df.drop(['Director'], axis=1)
    
    # Handle Actor popularity (use default values for new actors)
    df['Actor1_popularity'] = 1  # Default for new actors
    df['Actor2_popularity'] = 1
    df['Actor3_popularity'] = 1
    
    df = df.drop(['Actor 1', 'Actor 2', 'Actor 3'], axis=1)
    
    # Create additional features
    df['Movie_Age'] = 2024 - df['Year']
    df['Total_Actor_Popularity'] = (
        df['Actor1_popularity'] + 
        df['Actor2_popularity'] + 
        df['Actor3_popularity']
    )
    
    # Select and order features
    # Add missing columns with default values
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_columns]
    
    # Scale numerical features
    numerical_cols = ['Year', 'Duration_min', 'Votes', 'Director_encoded', 
                     'Actor1_popularity', 'Actor2_popularity', 'Actor3_popularity',
                     'Movie_Age', 'Total_Actor_Popularity']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

def predict_rating(movie_data):
    """Predict rating for a movie."""
    # Load model and preprocessing
    model, preprocessing = load_model_and_preprocessing()
    
    # Preprocess input
    X = preprocess_input(movie_data, preprocessing)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction

def main():
    """Example usage of the prediction function."""
    print("=" * 60)
    print("MOVIE RATING PREDICTION")
    print("=" * 60)
    
    # Example movie data
    example_movie = {
        'Year': 2023,
        'Duration': '120 min',
        'Genre': 'Drama, Action',
        'Votes': '1000',
        'Director': 'Christopher Nolan',
        'Actor 1': 'Cillian Murphy',
        'Actor 2': 'Emily Blunt',
        'Actor 3': 'Matt Damon'
    }
    
    print("\nExample Movie:")
    for key, value in example_movie.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    prediction = predict_rating(example_movie)
    
    print("\nPrediction Result:")
    print(f"  Predicted Rating: {prediction:.2f}/10")
    
    print("\n" + "=" * 60)
    print("To make predictions for your own data:")
    print("1. Import the predict_rating function")
    print("2. Pass a dictionary with movie information")
    print("3. Required fields: Year, Duration, Genre, Votes, Director, Actor 1, Actor 2, Actor 3")
    print("=" * 60)

if __name__ == "__main__":
    main()
