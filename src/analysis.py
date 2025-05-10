import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os


# Ensure directories exist
os.makedirs('../visualizations', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# Load the dataset
data_path = '../data/2019.csv'
df = pd.read_csv(data_path)

# Data Cleaning
# Rename columns for consistency and easier access
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Handle missing values by dropping rows with any missing data
df.dropna(inplace=True)

# Ensure appropriate data types (example: convert numeric columns if needed)
numeric_columns = ['ladder_score', 'logged_gdp_per_capita', 'social_support',
                  'healthy_life_expectancy', 'freedom_to_make_life_choices',
                  'generosity', 'perceptions_of_corruption']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Exploratory Data Analysis (EDA)
# Summary statistics
print("Summary Statistics:")
print(df.describe())

# Correlation matrix
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Happiness Factors')
plt.savefig('../visualizations/correlation_matrix.png')
plt.close()

# Scatter plot: GDP vs Happiness Score
plt.figure(figsize=(8, 6))
sns.scatterplot(x='logged_gdp_per_capita', y='ladder_score', data=df)
plt.title('Logged GDP per Capita vs Happiness Score')
plt.xlabel('Logged GDP per Capita')
plt.ylabel('Happiness Score')
plt.savefig('../visualizations/gdp_vs_happiness.png')
plt.close()

# Bar chart: Average happiness score by region
# Note: Assuming the dataset has a 'Regional indicator' column; adjust if named differently
if 'regional_indicator' in df.columns:
    regional_avg = df.groupby('regional_indicator')['ladder_score'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    regional_avg.plot(kind='barh', color='skyblue')
    plt.title('Average Happiness Score by Region')
    plt.xlabel('Average Happiness Score')
    plt.ylabel('Region')
    plt.savefig('../visualizations/regional_happiness.png')
    plt.close()
else:
    print("No 'regional_indicator' column found in the dataset.")

# Advanced Analysis: Feature Importance with Linear Regression
# Prepare data for modeling
features = ['logged_gdp_per_capita', 'social_support', 'healthy_life_expectancy',
           'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption']
X = df[features]
y = df['ladder_score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Display feature importance
print("\nFeature Importance (Linear Regression Coefficients):")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.4f}")

# Save the model
joblib.dump(model, '../models/linear_regression_model.pkl')