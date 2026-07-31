import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data(filepath):
    """Load the IMDb movie dataset and perform initial exploration."""
    print("=" * 60)
    print("MOVIE RATING PREDICTION MODEL")
    print("=" * 60)
    
    # Load data with proper header
    df = pd.read_excel(filepath, header=1)
    print(f"\nDataset loaded successfully!")
    print(f"Total movies: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic info
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(df.info())
    
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())
    
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print(df.describe())
    
    # Rating statistics
    print(f"\nRating Statistics:")
    print(f"  Mean: {df['Rating'].mean():.2f}")
    print(f"  Median: {df['Rating'].median():.2f}")
    print(f"  Std: {df['Rating'].std():.2f}")
    print(f"  Min: {df['Rating'].min():.2f}")
    print(f"  Max: {df['Rating'].max():.2f}")
    
    return df

def visualize_data(df):
    """Create visualizations to understand the data better."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Rating distribution
    sns.histplot(data=df, x='Rating', bins=20, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Rating Distribution')
    axes[0, 0].set_xlabel('Rating')
    
    # Year vs Rating
    df_year = df[df['Year'] > 0]
    sns.scatterplot(data=df_year, x='Year', y='Rating', alpha=0.5, ax=axes[0, 1])
    axes[0, 1].set_title('Year vs Rating')
    axes[0, 1].set_xlabel('Year')
    
    # Top genres
    genre_counts = df['Genre'].str.split(', ', expand=True).stack().value_counts().head(10)
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=axes[0, 2])
    axes[0, 2].set_title('Top 10 Genres')
    axes[0, 2].set_xlabel('Count')
    
    # Duration vs Rating
    df_duration = df[df['Duration'].notna()]
    # Extract duration in minutes
    df_duration['Duration_min'] = df_duration['Duration'].str.extract(r'(\d+)').astype(float)
    sns.scatterplot(data=df_duration, x='Duration_min', y='Rating', alpha=0.5, ax=axes[1, 0])
    axes[1, 0].set_title('Duration vs Rating')
    axes[1, 0].set_xlabel('Duration (minutes)')
    
    # Votes distribution (log scale)
    df_votes = df[df['Votes'].notna()]
    try:
        df_votes['Votes_num'] = pd.to_numeric(df_votes['Votes'].str.replace(',', ''), errors='coerce')
        sns.histplot(data=df_votes, x='Votes_num', bins=30, ax=axes[1, 1])
        axes[1, 1].set_title('Votes Distribution')
        axes[1, 1].set_xlabel('Number of Votes')
        axes[1, 1].set_yscale('log')
    except:
        axes[1, 1].text(0.5, 0.5, 'Votes data unavailable', ha='center', va='center')
    
    # Top directors by movie count
    director_counts = df['Director'].value_counts().head(10)
    sns.barplot(x=director_counts.values, y=director_counts.index, ax=axes[1, 2])
    axes[1, 2].set_title('Top 10 Directors by Movie Count')
    axes[1, 2].set_xlabel('Number of Movies')
    
    plt.tight_layout()
    plt.savefig('/Users/maharshipatel/Downloads/codsfot/imdb/movie_visualizations.png', dpi=150, bbox_inches='tight')
    print("Visualizations saved as 'movie_visualizations.png'")
    plt.close()

def preprocess_data(df):
    """Preprocess the data for modeling."""
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Drop rows where Rating is missing (target variable)
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna(subset=['Rating'])
    print(f"Dropped {initial_rows - len(df_processed)} rows with missing ratings")
    
    # Drop Name column (not useful for prediction)
    df_processed = df_processed.drop(['Name'], axis=1)
    print("Dropped Name column")
    
    # Handle Year - fill missing with median
    df_processed['Year'] = pd.to_numeric(df_processed['Year'], errors='coerce')
    median_year = df_processed['Year'].median()
    df_processed['Year'] = df_processed['Year'].fillna(median_year)
    print(f"Filled missing Year values with median: {median_year:.0f}")
    
    # Handle Duration - extract minutes and fill missing
    df_processed['Duration_min'] = df_processed['Duration'].str.extract(r'(\d+)').astype(float)
    median_duration = df_processed['Duration_min'].median()
    df_processed['Duration_min'] = df_processed['Duration_min'].fillna(median_duration)
    print(f"Extracted duration in minutes, filled missing with median: {median_duration:.0f}")
    df_processed = df_processed.drop(['Duration'], axis=1)
    
    # Handle Votes - convert to numeric and fill missing
    df_processed['Votes'] = df_processed['Votes'].astype(str).str.replace(',', '')
    df_processed['Votes'] = pd.to_numeric(df_processed['Votes'], errors='coerce')
    median_votes = df_processed['Votes'].median()
    df_processed['Votes'] = df_processed['Votes'].fillna(median_votes)
    print(f"Converted Votes to numeric, filled missing with median: {median_votes:.0f}")
    
    # Handle Genre - create binary features for top genres
    print("Creating genre features...")
    all_genres = df_processed['Genre'].str.split(', ', expand=True).stack().value_counts()
    top_genres = all_genres.head(15).index.tolist()
    
    for genre in top_genres:
        df_processed[f'Genre_{genre}'] = df_processed['Genre'].apply(lambda x: 1 if genre in str(x) else 0)
    
    df_processed = df_processed.drop(['Genre'], axis=1)
    print(f"Created binary features for top 15 genres")
    
    # Handle Director - encode top directors
    print("Encoding director features...")
    director_counts = df_processed['Director'].value_counts()
    top_directors = director_counts[director_counts >= 5].index.tolist()
    
    # Create director encoding
    df_processed['Director_encoded'] = df_processed['Director'].apply(
        lambda x: x if x in top_directors else 'Other'
    )
    
    le_director = LabelEncoder()
    df_processed['Director_encoded'] = le_director.fit_transform(df_processed['Director_encoded'])
    df_processed = df_processed.drop(['Director'], axis=1)
    print(f"Encoded directors (top {len(top_directors)} directors + 'Other')")
    
    # Handle Actors - create actor popularity features
    print("Creating actor features...")
    # Actor 1
    actor1_counts = df_processed['Actor 1'].value_counts()
    df_processed['Actor1_popularity'] = df_processed['Actor 1'].map(actor1_counts).fillna(0)
    
    # Actor 2
    actor2_counts = df_processed['Actor 2'].value_counts()
    df_processed['Actor2_popularity'] = df_processed['Actor 2'].map(actor2_counts).fillna(0)
    
    # Actor 3
    actor3_counts = df_processed['Actor 3'].value_counts()
    df_processed['Actor3_popularity'] = df_processed['Actor 3'].map(actor3_counts).fillna(0)
    
    df_processed = df_processed.drop(['Actor 1', 'Actor 2', 'Actor 3'], axis=1)
    print("Created actor popularity features")
    
    # Create additional features
    df_processed['Movie_Age'] = 2024 - df_processed['Year']
    print("Created Movie_Age feature")
    
    df_processed['Total_Actor_Popularity'] = (
        df_processed['Actor1_popularity'] + 
        df_processed['Actor2_popularity'] + 
        df_processed['Actor3_popularity']
    )
    print("Created Total_Actor_Popularity feature")
    
    print(f"\nFinal dataset shape: {df_processed.shape}")
    print(f"Final columns: {list(df_processed.columns)}")
    
    return df_processed, le_director, top_genres

def build_and_train_model(X_train, y_train):
    """Build and train the regression model."""
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    # Try multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_score = -float('inf')
    best_model_name = ""
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        print(f"\n{name}:")
        print(f"  Cross-validation R²: {mean_score:.4f} (+/- {std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_model_name = name
    
    # Train the best model on full training data
    best_model.fit(X_train, y_train)
    print(f"\nBest model: {best_model_name} with R²: {best_score:.4f}")
    
    return best_model, best_model_name

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the model performance."""
    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title(f'Actual vs Predicted Ratings - {model_name}')
    plt.tight_layout()
    plt.savefig('/Users/maharshipatel/Downloads/codsfot/imdb/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    print("Actual vs Predicted plot saved as 'actual_vs_predicted.png'")
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {model_name}')
    plt.tight_layout()
    plt.savefig('/Users/maharshipatel/Downloads/codsfot/imdb/residual_plot.png', dpi=150, bbox_inches='tight')
    print("Residual plot saved as 'residual_plot.png'")
    plt.close()
    
    # Feature importance (for Random Forest)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Feature Importance:")
        print(feature_importance.head(15))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('/Users/maharshipatel/Downloads/codsfot/imdb/feature_importance.png', dpi=150, bbox_inches='tight')
        print("Feature importance plot saved as 'feature_importance.png'")
        plt.close()
    
    return r2, rmse, mae

def save_model(model, scaler, le_director, top_genres, feature_columns):
    """Save the trained model and preprocessing objects."""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save model
    joblib.dump(model, '/Users/maharshipatel/Downloads/codsfot/imdb/movie_rating_model.pkl')
    print("Model saved as 'movie_rating_model.pkl'")
    
    # Save preprocessing objects
    preprocessing = {
        'scaler': scaler,
        'le_director': le_director,
        'top_genres': top_genres,
        'feature_columns': feature_columns
    }
    joblib.dump(preprocessing, '/Users/maharshipatel/Downloads/codsfot/imdb/preprocessing.pkl')
    print("Preprocessing objects saved as 'preprocessing.pkl'")

def main():
    """Main function to run the entire pipeline."""
    # Load and explore data
    df = load_and_explore_data('/Users/maharshipatel/Downloads/codsfot/imdb/IMDb Movies India.xlsx')
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    df_processed, le_director, top_genres = preprocess_data(df)
    
    # Split features and target
    X = df_processed.drop('Rating', axis=1)
    y = df_processed['Rating']
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Year', 'Duration_min', 'Votes', 'Director_encoded', 
                     'Actor1_popularity', 'Actor2_popularity', 'Actor3_popularity',
                     'Movie_Age', 'Total_Actor_Popularity']
    
    # Only scale columns that exist
    numerical_cols = [col for col in numerical_cols if col in X.columns]
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Build and train model
    model, model_name = build_and_train_model(X_train, y_train)
    
    # Evaluate model
    r2, rmse, mae = evaluate_model(model, X_test, y_test, model_name)
    
    # Save model
    save_model(model, scaler, le_director, top_genres, X.columns.tolist())
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final R² Score: {r2:.4f}")
    print(f"Final RMSE: {rmse:.4f}")
    print(f"Final MAE: {mae:.4f}")
    print("\nFiles generated:")
    print("  - movie_visualizations.png")
    print("  - actual_vs_predicted.png")
    print("  - residual_plot.png")
    print("  - feature_importance.png")
    print("  - movie_rating_model.pkl")
    print("  - preprocessing.pkl")

if __name__ == "__main__":
    main()
