# Movie Rating Prediction Model

A machine learning model that predicts movie ratings based on features like genre, director, actors, and other movie characteristics.

## Project Overview

This project uses the IMDb Movies India dataset to build a regression model that predicts movie ratings (on a scale of 1-10). The model analyzes historical movie data to identify factors that influence ratings and can estimate ratings for new movies.

## Dataset

The dataset contains 15,509 Indian movies with the following features:
- **Name**: Movie title
- **Year**: Release year
- **Duration**: Movie duration in minutes
- **Genre**: Movie genre(s) - can be multiple
- **Rating**: Target variable (1-10 scale)
- **Votes**: Number of user votes
- **Director**: Movie director
- **Actor 1**: Primary actor
- **Actor 2**: Secondary actor
- **Actor 3**: Tertiary actor

## Model Performance

The best performing model is **Random Forest Regressor** with:
- **Cross-validation R²**: 0.3682
- **Test R² Score**: 0.3767
- **Root Mean Squared Error (RMSE)**: 1.0765
- **Mean Absolute Error (MAE)**: 0.8133

## Features Used

The model uses the following engineered features:

### Original Features
- Year (Release year)
- Duration_min (Duration in minutes)
- Votes (Number of votes)

### Genre Features (Binary)
- Top 15 genres: Drama, Action, Romance, Comedy, Crime, Thriller, Family, Musical, Adventure, Mystery, Horror, Fantasy, Documentary, Biography, History

### Director Features
- Director_encoded (Label-encoded for top directors)

### Actor Features
- Actor1_popularity (Number of movies by Actor 1)
- Actor2_popularity (Number of movies by Actor 2)
- Actor3_popularity (Number of movies by Actor 3)

### Engineered Features
- Movie_Age (2024 - Year)
- Total_Actor_Popularity (Sum of all actor popularities)

## Key Findings

### Top Feature Importance
1. **Votes** (23.9%) - Number of votes is the strongest predictor
2. **Year** (11.0%) - Release year influences ratings
3. **Total_Actor_Popularity** (11.0%) - Combined actor popularity
4. **Movie_Age** (9.8%) - Older movies tend to have different ratings
5. **Duration_min** (9.2%) - Movie length impacts ratings

### Data Insights
- Average rating: 5.84/10
- Median rating: 6.00/10
- Rating range: 1.10 to 10.00
- 7,919 movies with complete rating data used for training
- Drama is the most common genre

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model from scratch:
```bash
python3 movie_rating_model.py
```

This will:
- Load and explore the dataset
- Create visualizations
- Preprocess the data
- Train and evaluate models
- Save the best model and preprocessing objects
- Generate performance plots

### Making Predictions

To predict ratings for new movies:
```bash
python3 predict_rating.py
```

Or use the prediction function in your own code:
```python
from predict_rating import predict_rating

movie_data = {
    'Year': 2023,
    'Duration': '120 min',
    'Genre': 'Drama, Action',
    'Votes': '1000',
    'Director': 'Christopher Nolan',
    'Actor 1': 'Cillian Murphy',
    'Actor 2': 'Emily Blunt',
    'Actor 3': 'Matt Damon'
}

prediction = predict_rating(movie_data)
print(f"Predicted Rating: {prediction:.2f}/10")
```

## Files Generated

- **movie_rating_model.pkl**: Trained Random Forest model
- **preprocessing.pkl**: Preprocessing objects (scaler, encoders, feature lists)
- **movie_visualizations.png**: Data exploration plots
- **actual_vs_predicted.png**: Actual vs predicted ratings scatter plot
- **residual_plot.png**: Residual analysis plot
- **feature_importance.png**: Feature importance visualization

## Model Architecture

The project compares two regression models:
1. **Linear Regression**: Baseline model
2. **Random Forest Regressor**: 100 estimators, best performer

Random Forest performed significantly better due to its ability to capture non-linear relationships in the data.

## Data Preprocessing

- Dropped rows with missing ratings (target variable)
- Dropped Name column (not useful for prediction)
- Filled missing Year values with median (-1997)
- Extracted duration in minutes from string format
- Converted Votes to numeric, filled missing with median (55)
- Created binary features for top 15 genres
- Encoded directors (top 375 directors + 'Other' category)
- Created actor popularity features based on movie count
- Engineered Movie_Age and Total_Actor_Popularity features
- Standardized numerical features using StandardScaler

## Model Evaluation Metrics

- **R² Score**: Measures proportion of variance explained (0.3767 = 37.67%)
- **RMSE**: Root Mean Squared Error (1.0765 rating points)
- **MAE**: Mean Absolute Error (0.8133 rating points)

## Limitations

- Model explains ~37% of rating variance - there are many other factors influencing ratings
- Actor popularity is based on movie count in dataset, not actual fame
- Director encoding limited to directors seen in training data
- Genre features are binary and don't capture genre combinations well
- Dataset is specific to Indian movies, may not generalize to other regions

## Future Improvements

- Add more features: budget, production company, release date, language
- Use advanced models: XGBoost, LightGBM, Neural Networks
- Perform hyperparameter tuning with GridSearch or RandomizedSearch
- Incorporate text features from movie descriptions/reviews
- Use ensemble methods to combine multiple models
- Add cross-validation with stratified splits
- Implement feature selection techniques
- Gather more diverse training data

## Technical Details

- **Language**: Python 3
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **Data Size**: 7,919 movies (after preprocessing)
- **Feature Count**: 25 features
- **Training/Test Split**: 80/20
