import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data(filepath):
    """Load the Titanic dataset and perform initial exploration."""
    print("=" * 50)
    print("TITANIC SURVIVAL PREDICTION MODEL")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\nDataset loaded successfully!")
    print(f"Total passengers: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic info
    print("\n" + "=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(df.info())
    
    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    print(df.isnull().sum())
    
    print("\n" + "=" * 50)
    print("STATISTICAL SUMMARY")
    print("=" * 50)
    print(df.describe())
    
    # Survival rate
    survival_rate = df['Survived'].mean() * 100
    print(f"\nOverall Survival Rate: {survival_rate:.2f}%")
    
    return df

def visualize_data(df):
    """Create visualizations to understand the data better."""
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Survival count
    sns.countplot(data=df, x='Survived', ax=axes[0, 0])
    axes[0, 0].set_title('Survival Count')
    axes[0, 0].set_xticklabels(['Did Not Survive', 'Survived'])
    
    # Survival by Sex
    sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0, 1])
    axes[0, 1].set_title('Survival by Sex')
    
    # Survival by Passenger Class
    sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[0, 2])
    axes[0, 2].set_title('Survival by Passenger Class')
    
    # Age distribution
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', ax=axes[1, 0])
    axes[1, 0].set_title('Age Distribution by Survival')
    
    # Fare distribution
    sns.histplot(data=df, x='Fare', hue='Survived', multiple='stack', ax=axes[1, 1])
    axes[1, 1].set_title('Fare Distribution by Survival')
    
    # Survival by Embarkation
    sns.countplot(data=df, x='Embarked', hue='Survived', ax=axes[1, 2])
    axes[1, 2].set_title('Survival by Embarkation Port')
    
    plt.tight_layout()
    plt.savefig('/Users/maharshipatel/Downloads/codsfot/titanic/titanic_visualizations.png', dpi=150, bbox_inches='tight')
    print("Visualizations saved as 'titanic_visualizations.png'")
    plt.close()

def preprocess_data(df):
    """Preprocess the data for modeling."""
    print("\n" + "=" * 50)
    print("PREPROCESSING DATA")
    print("=" * 50)
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Drop columns that won't be useful for prediction
    df_processed = df_processed.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    print("Dropped columns: PassengerId, Name, Ticket, Cabin")
    
    # Handle missing values
    # Fill missing Age with median
    df_processed['Age'] = df_processed['Age'].fillna(df_processed['Age'].median())
    print("Filled missing Age values with median")
    
    # Fill missing Embarked with mode
    df_processed['Embarked'] = df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0])
    print("Filled missing Embarked values with mode")
    
    # Fill missing Fare with median (if any)
    df_processed['Fare'] = df_processed['Fare'].fillna(df_processed['Fare'].median())
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    df_processed['Sex'] = le_sex.fit_transform(df_processed['Sex'])
    print("Encoded Sex variable")
    
    le_embarked = LabelEncoder()
    df_processed['Embarked'] = le_embarked.fit_transform(df_processed['Embarked'])
    print("Encoded Embarked variable")
    
    # Create family size feature
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    print("Created FamilySize feature")
    
    # Create IsAlone feature
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
    print("Created IsAlone feature")
    
    print(f"\nFinal dataset shape: {df_processed.shape}")
    print(f"Final columns: {list(df_processed.columns)}")
    
    return df_processed, le_sex, le_embarked

def build_and_train_model(X_train, y_train):
    """Build and train the classification model."""
    print("\n" + "=" * 50)
    print("TRAINING MODELS")
    print("=" * 50)
    
    # Try multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        print(f"\n{name}:")
        print(f"  Cross-validation accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_model_name = name
    
    # Train the best model on full training data
    best_model.fit(X_train, y_train)
    print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")
    
    return best_model, best_model_name

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate the model performance."""
    print("\n" + "=" * 50)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Did Not Survive', 'Survived'],
                yticklabels=['Did Not Survive', 'Survived'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/Users/maharshipatel/Downloads/codsfot/titanic/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
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
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('/Users/maharshipatel/Downloads/codsfot/titanic/feature_importance.png', dpi=150, bbox_inches='tight')
        print("Feature importance plot saved as 'feature_importance.png'")
        plt.close()
    
    return accuracy

def save_model(model, scaler, le_sex, le_embarked, feature_columns):
    """Save the trained model and preprocessing objects."""
    print("\n" + "=" * 50)
    print("SAVING MODEL")
    print("=" * 50)
    
    # Save model
    joblib.dump(model, '/Users/maharshipatel/Downloads/codsfot/titanic/titanic_model.pkl')
    print("Model saved as 'titanic_model.pkl'")
    
    # Save preprocessing objects
    preprocessing = {
        'scaler': scaler,
        'le_sex': le_sex,
        'le_embarked': le_embarked,
        'feature_columns': feature_columns
    }
    joblib.dump(preprocessing, '/Users/maharshipatel/Downloads/codsfot/titanic/preprocessing.pkl')
    print("Preprocessing objects saved as 'preprocessing.pkl'")

def main():
    """Main function to run the entire pipeline."""
    # Load and explore data
    df = load_and_explore_data('/Users/maharshipatel/Downloads/codsfot/titanic/Titanic-Dataset.csv.xls')
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    df_processed, le_sex, le_embarked = preprocess_data(df)
    
    # Split features and target
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Build and train model
    model, model_name = build_and_train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test, model_name)
    
    # Save model
    save_model(model, scaler, le_sex, le_embarked, X.columns.tolist())
    
    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("\nFiles generated:")
    print("  - titanic_visualizations.png")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - titanic_model.pkl")
    print("  - preprocessing.pkl")

if __name__ == "__main__":
    main()
