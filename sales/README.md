# Sales Prediction Model

A machine learning model that predicts product sales based on advertising expenditure across different media channels (TV, Radio, Newspaper).

## Project Overview

This project uses advertising data to build a regression model that predicts sales based on advertising spend. The model helps businesses optimize their advertising strategies by understanding the relationship between advertising expenditure and sales performance.

## Dataset

The dataset contains 200 records of advertising spend and corresponding sales:
- **TV**: Advertising budget for TV (in thousands of dollars)
- **Radio**: Advertising budget for Radio (in thousands of dollars)
- **Newspaper**: Advertising budget for Newspaper (in thousands of dollars)
- **Sales**: Product sales (in thousands of units)

### Data Statistics
- **Total Records**: 200
- **Average Sales**: 15.13k units
- **Sales Range**: 1.60k to 27.00k units
- **No Missing Values**: Clean dataset

## Model Performance

The best performing model is **Random Forest Regressor** with:
- **Cross-validation R²**: 0.9403
- **Test R² Score**: 0.9526
- **Root Mean Squared Error (RMSE)**: 1.2105
- **Mean Absolute Error (MAE)**: 0.8327

## Features Used

### Original Features
- TV (TV advertising spend)
- Radio (Radio advertising spend)
- Newspaper (Newspaper advertising spend)

### Engineered Features
- **Total_Ad_Spend**: Sum of all advertising channels
- **TV_Radio_Ratio**: TV spend divided by Radio spend
- **TV_Newspaper_Ratio**: TV spend divided by Newspaper spend
- **Radio_Newspaper_Ratio**: Radio spend divided by Newspaper spend
- **TV_Radio_Interaction**: Product of TV and Radio spend
- **TV_Newspaper_Interaction**: Product of TV and Newspaper spend
- **Radio_Newspaper_Interaction**: Product of Radio and Newspaper spend

## Key Findings

### Correlation Analysis
- **TV**: 0.901 correlation with Sales (strongest predictor)
- **Radio**: 0.350 correlation with Sales (moderate predictor)
- **Newspaper**: 0.158 correlation with Sales (weak predictor)

### Feature Importance
1. **Total_Ad_Spend** (69.7%) - Total advertising budget is the most important factor
2. **TV_Radio_Interaction** (15.3%) - Combined TV and Radio spend
3. **TV** (11.5%) - Individual TV advertising spend
4. **TV_Newspaper_Ratio** (0.6%) - Ratio of TV to Newspaper spend
5. **TV_Newspaper_Interaction** (0.6%) - Combined TV and Newspaper spend

### Business Insights
- TV advertising has the strongest impact on sales
- Total advertising budget is the most critical factor
- Interaction between TV and Radio advertising shows synergistic effects
- Newspaper advertising has minimal impact on sales
- Balanced advertising across channels may not be optimal - TV-focused strategies perform better

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model from scratch:
```bash
python3 sales_prediction_model.py
```

This will:
- Load and explore the dataset
- Create visualizations
- Preprocess the data and engineer features
- Train and evaluate models
- Save the best model and preprocessing objects
- Generate performance plots

### Making Predictions

To predict sales for new advertising budgets:
```bash
python3 predict_sales.py
```

Or use the prediction function in your own code:
```python
from predict_sales import predict_sales

ad_data = {
    'TV': 200,      # TV advertising spend in thousands
    'Radio': 30,    # Radio advertising spend in thousands
    'Newspaper': 40 # Newspaper advertising spend in thousands
}

prediction = predict_sales(ad_data)
print(f"Predicted Sales: {prediction:.2f}k units")
```

### Optimization Scenarios

The prediction script includes several budget allocation scenarios:
- **High Budget**: $340k total - Predicted 23.44k sales
- **Medium Budget**: $205k total - Predicted 16.70k sales
- **Low Budget**: $140k total - Predicted 12.03k sales
- **TV Focused**: $320k total (mostly TV) - Predicted 19.62k sales
- **Balanced**: $150k total (equal distribution) - Predicted 12.69k sales

## Files Generated

- **sales_model.pkl**: Trained Random Forest model
- **preprocessing.pkl**: Preprocessing objects (scaler, feature list)
- **sales_visualizations.png**: Data exploration plots
- **actual_vs_predicted.png**: Actual vs predicted sales scatter plot
- **residual_plot.png**: Residual analysis plot
- **feature_importance.png**: Feature importance visualization
- **coefficients.png**: Linear regression coefficients (if applicable)

## Model Architecture

The project compares two regression models:
1. **Linear Regression**: Baseline model with interpretable coefficients
2. **Random Forest Regressor**: 100 estimators, best performer

Random Forest performed significantly better due to its ability to capture non-linear relationships and feature interactions.

## Data Preprocessing

- No missing values to handle (clean dataset)
- Created 7 engineered features from original 3 features
- Standardized all features using StandardScaler
- Feature engineering focused on:
  - Total spend
  - Ratios between channels
  - Interaction effects between channels

## Model Evaluation Metrics

- **R² Score**: 0.9526 (95.26% of variance explained)
- **RMSE**: 1.2105 (average prediction error of 1.21k units)
- **MAE**: 0.8327 (average absolute error of 0.83k units)

## Business Applications

### Budget Optimization
- Use the model to test different budget allocation scenarios
- Identify the most cost-effective advertising mix
- Maximize ROI by focusing on high-impact channels

### Sales Forecasting
- Predict sales based on planned advertising budgets
- Set realistic sales targets
- Plan inventory and resources accordingly

### Strategy Planning
- Understand which advertising channels drive sales
- Make data-driven decisions about advertising spend
- Optimize marketing budget allocation

## Limitations

- Model trained on historical data - may not account for market changes
- Limited to three advertising channels
- Does not consider seasonality or external factors
- Assumes linear relationship between budget and sales (within range)
- Small dataset (200 records) - may not capture all variations

## Future Improvements

- Add more advertising channels (social media, digital, etc.)
- Include temporal features (seasonality, trends)
- Add external factors (economic indicators, competitor data)
- Use time series models for forecasting
- Implement hyperparameter tuning
- Gather more training data
- Add A/B testing capabilities
- Create interactive dashboard for scenario planning

## Technical Details

- **Language**: Python 3
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **Data Size**: 200 records
- **Feature Count**: 10 features (3 original + 7 engineered)
- **Training/Test Split**: 80/20
- **Cross-validation**: 5-fold

## Example Use Cases

1. **Campaign Planning**: "If we spend $250k on TV, $40k on Radio, and $50k on Newspaper, what sales can we expect?"
2. **Budget Allocation**: "How should we distribute our $300k budget across channels to maximize sales?"
3. **ROI Analysis**: "Which advertising channel gives us the best return on investment?"
4. **Sales Targeting**: "What advertising budget do we need to achieve 20k in sales?"
