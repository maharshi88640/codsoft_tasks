import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
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
    """Load the advertising dataset and perform initial exploration."""
    print("=" * 60)
    print("SALES PREDICTION MODEL")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\nDataset loaded successfully!")
    print(f"Total records: {len(df)}")
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
    
    # Sales statistics
    print(f"\nSales Statistics:")
    print(f"  Mean: {df['Sales'].mean():.2f}")
    print(f"  Median: {df['Sales'].median():.2f}")
    print(f"  Std: {df['Sales'].std():.2f}")
    print(f"  Min: {df['Sales'].min():.2f}")
    print(f"  Max: {df['Sales'].max():.2f}")
    
    # Correlation analysis
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    correlation = df.corr()
    print(correlation['Sales'].sort_values(ascending=False))
    
    return df

def visualize_data(df):
    """Create visualizations to understand the data better."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Sales distribution
    sns.histplot(data=df, x='Sales', bins=20, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Sales Distribution')
    axes[0, 0].set_xlabel('Sales (in thousands)')
    
    # TV vs Sales
    sns.scatterplot(data=df, x='TV', y='Sales', alpha=0.6, ax=axes[0, 1])
    axes[0, 1].set_title('TV Advertising vs Sales')
    axes[0, 1].set_xlabel('TV Advertising (in thousands)')
    axes[0, 1].set_ylabel('Sales (in thousands)')
    
    # Radio vs Sales
    sns.scatterplot(data=df, x='Radio', y='Sales', alpha=0.6, ax=axes[0, 2])
    axes[0, 2].set_title('Radio Advertising vs Sales')
    axes[0, 2].set_xlabel('Radio Advertising (in thousands)')
    axes[0, 2].set_ylabel('Sales (in thousands)')
    
    # Newspaper vs Sales
    sns.scatterplot(data=df, x='Newspaper', y='Sales', alpha=0.6, ax=axes[1, 0])
    axes[1, 0].set_title('Newspaper Advertising vs Sales')
    axes[1, 0].set_xlabel('Newspaper Advertising (in thousands)')
    axes[1, 0].set_ylabel('Sales (in thousands)')
    
    # Advertising spend distribution
    ad_spend = df[['TV', 'Radio', 'Newspaper']].sum()
    sns.barplot(x=ad_spend.index, y=ad_spend.values, ax=axes[1, 1])
    axes[1, 1].set_title('Total Advertising Spend by Channel')
    axes[1, 1].set_ylabel('Total Spend (in thousands)')
    
    # Correlation heatmap
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
    axes[1, 2].set_title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('/Users/maharshipatel/Downloads/codsfot/sales/sales_visualizations.png', dpi=150, bbox_inches='tight')
    print("Visualizations saved as 'sales_visualizations.png'")
    plt.close()

def preprocess_data(df):
    """Preprocess the data for modeling."""
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # No missing values to handle
    print("No missing values found in dataset")
    
    # Create additional features
    df_processed['Total_Ad_Spend'] = df_processed['TV'] + df_processed['Radio'] + df_processed['Newspaper']
    print("Created Total_Ad_Spend feature")
    
    df_processed['TV_Radio_Ratio'] = df_processed['TV'] / (df_processed['Radio'] + 1)
    print("Created TV_Radio_Ratio feature")
    
    df_processed['TV_Newspaper_Ratio'] = df_processed['TV'] / (df_processed['Newspaper'] + 1)
    print("Created TV_Newspaper_Ratio feature")
    
    df_processed['Radio_Newspaper_Ratio'] = df_processed['Radio'] / (df_processed['Newspaper'] + 1)
    print("Created Radio_Newspaper_Ratio feature")
    
    # Create interaction features
    df_processed['TV_Radio_Interaction'] = df_processed['TV'] * df_processed['Radio']
    print("Created TV_Radio_Interaction feature")
    
    df_processed['TV_Newspaper_Interaction'] = df_processed['TV'] * df_processed['Newspaper']
    print("Created TV_Newspaper_Interaction feature")
    
    df_processed['Radio_Newspaper_Interaction'] = df_processed['Radio'] * df_processed['Newspaper']
    print("Created Radio_Newspaper_Interaction feature")
    
    print(f"\nFinal dataset shape: {df_processed.shape}")
    print(f"Final columns: {list(df_processed.columns)}")
    
    return df_processed

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
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(f'Actual vs Predicted Sales - {model_name}')
    plt.tight_layout()
    plt.savefig('/Users/maharshipatel/Downloads/codsfot/sales/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    print("Actual vs Predicted plot saved as 'actual_vs_predicted.png'")
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {model_name}')
    plt.tight_layout()
    plt.savefig('/Users/maharshipatel/Downloads/codsfot/sales/residual_plot.png', dpi=150, bbox_inches='tight')
    print("Residual plot saved as 'residual_plot.png'")
    plt.close()
    
    # Feature importance (for Random Forest)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('/Users/maharshipatel/Downloads/codsfot/sales/feature_importance.png', dpi=150, bbox_inches='tight')
        print("Feature importance plot saved as 'feature_importance.png'")
        plt.close()
    
    # Coefficients (for Linear Regression)
    if hasattr(model, 'coef_'):
        coefficients = pd.DataFrame({
            'feature': X_test.columns,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nModel Coefficients:")
        print(coefficients)
        
        # Plot coefficients
        plt.figure(figsize=(12, 8))
        sns.barplot(data=coefficients, x='coefficient', y='feature')
        plt.title('Model Coefficients')
        plt.xlabel('Coefficient Value')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig('/Users/maharshipatel/Downloads/codsfot/sales/coefficients.png', dpi=150, bbox_inches='tight')
        print("Coefficients plot saved as 'coefficients.png'")
        plt.close()
    
    return r2, rmse, mae

def save_model(model, scaler, feature_columns):
    """Save the trained model and preprocessing objects."""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save model
    joblib.dump(model, '/Users/maharshipatel/Downloads/codsfot/sales/sales_model.pkl')
    print("Model saved as 'sales_model.pkl'")
    
    # Save preprocessing objects
    preprocessing = {
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    joblib.dump(preprocessing, '/Users/maharshipatel/Downloads/codsfot/sales/preprocessing.pkl')
    print("Preprocessing objects saved as 'preprocessing.pkl'")

def main():
    """Main function to run the entire pipeline."""
    # Load and explore data
    df = load_and_explore_data('/Users/maharshipatel/Downloads/codsfot/sales/advertising.csv.xls')
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Split features and target
    X = df_processed.drop('Sales', axis=1)
    y = df_processed['Sales']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Build and train model
    model, model_name = build_and_train_model(X_train, y_train)
    
    # Evaluate model
    r2, rmse, mae = evaluate_model(model, X_test, y_test, model_name)
    
    # Save model
    save_model(model, scaler, X.columns.tolist())
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final R² Score: {r2:.4f}")
    print(f"Final RMSE: {rmse:.4f}")
    print(f"Final MAE: {mae:.4f}")
    print("\nFiles generated:")
    print("  - sales_visualizations.png")
    print("  - actual_vs_predicted.png")
    print("  - residual_plot.png")
    print("  - feature_importance.png")
    print("  - coefficients.png")
    print("  - sales_model.pkl")
    print("  - preprocessing.pkl")

if __name__ == "__main__":
    main()
