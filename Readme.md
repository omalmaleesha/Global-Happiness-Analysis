"""
# Analysis of Global Happiness Data

## Project Description
This project analyzes the World Happiness Report dataset (2020) to uncover insights into factors influencing global well-being. Using Python and free libraries like Pandas, Matplotlib, and Seaborn, we clean, analyze, and visualize the data to identify trends and correlations that can inform policies and personal decisions worldwide.

## Why It’s Valuable
- Provides insights into global happiness trends.
- Identifies key factors influencing well-being, such as GDP, social support, and freedom.
- Free to create using open-source tools and datasets.

## Setup and Installation
1. **Clone the Repository** (if hosted on GitHub):
   ```bash
   git clone https://github.com/yourusername/global_happiness_analysis.git
   ```
2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the Dataset**:
   - Get the World Happiness Report dataset (e.g., 2020.csv) from [Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness).
   - Place it in the `data/` folder as `2020.csv`.

## Running the Project
1. Navigate to the `src/` directory:
   ```bash
   cd src
   ```
2. Run the analysis script:
   ```bash
   python analysis.py
   ```
   - This will generate visualizations in the `visualizations/` folder and save a trained model in the `models/` folder.

## File Structure
```
global_happiness_analysis/
├── data/
│   └── 2020.csv              # World Happiness Report dataset
├── src/
│   └── analysis.py           # Main analysis script
├── visualizations/
│   └── (generated files)     # Correlation matrix, scatter plots, bar charts
├── models/
│   └── (generated files)     # Trained linear regression model
├── README.md                 # Project documentation
└── requirements.txt          # List of required Python libraries
```

## Code Explanation
- **Data Cleaning**: Loads the dataset, renames columns, handles missing values, and ensures correct data types.
- **Exploratory Data Analysis (EDA)**:
  - Generates summary statistics.
  - Creates a correlation matrix heatmap to show relationships between variables.
  - Plots a scatter plot of GDP vs. happiness score.
  - Displays a bar chart of average happiness scores by region.
- **Advanced Analysis**: Uses linear regression to determine the importance of factors influencing happiness scores.

## Insights and Conclusions
- **Correlations**: Strong positive correlations exist between happiness scores and factors like GDP per capita, social support, and life expectancy.
- **Regional Trends**: Happiness scores vary significantly by region, with some regions consistently scoring higher.
- **Feature Importance**: Factors like social support and GDP per capita tend to have the most significant impact on happiness scores (based on regression coefficients).

## Visualizations
Generated files in `visualizations/`:
- `correlation_matrix.png`: Heatmap of correlations between happiness factors.
- `gdp_vs_happiness.png`: Scatter plot of GDP per capita vs. happiness score.
- `regional_happiness.png`: Bar chart of average happiness scores by region.

## Requirements
- Python 3.x
- Libraries: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

## Notes
- Ensure the dataset file matches the expected column names (e.g., 'ladder_score' for happiness score). Adjust the code if column names differ.
- Visualizations are saved as PNG files for easy sharing and review.
"""