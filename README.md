# T20 Cricket Performance Analysis

## Overview
This project analyzes T20 cricket player performance using machine learning techniques. The dataset includes batting and bowling statistics, and we implement various supervised and unsupervised learning algorithms to predict performance, classify players, and identify key insights.

## Dataset
The analysis is performed on two datasets:
- **Batting Dataset (`t20.csv`)**: Contains batting statistics such as runs, strike rate, average, and balls faced.
- **Bowling Dataset (`Bowling_t20.csv`)**: Contains bowling statistics such as wickets, economy rate, and bowling average.

## Data Cleaning & Preprocessing
- Removed unnecessary columns (`Unnamed: 0`, `Span`, `Mat`, `Inns`).
- Converted string numerical values (e.g., `Runs` with `*` symbols) into integers.
- Converted missing and non-numeric values to NaN and filled them with the column mean.
- Removed invalid and null entries from key statistics such as `SR`, `Ave`, `BF`, and `Wkts`.

## Exploratory Data Analysis (EDA)
We analyzed key performance indicators using visualization techniques:
- **Bar Chart: Top 10 Batsmen** based on predicted averages.
- **Bar Chart: Top 10 Batsmen** based on actual strike rates.
- **Bar Chart: Top 10 Bowlers** based on wickets per match.
- **Pairplot: K-Means Clustering Results** to analyze player segmentation.

## Machine Learning Models

### Supervised Learning Algorithms
#### 1. **Linear Regression for Batting Performance Prediction**
- **Features**: Runs, Balls Faced, Strike Rate
- **Target**: Batting Average
- **Results**: 
    - Model was trained to predict batting average based on past performances.
    - Top 10 batsmen were ranked based on predicted average.

#### 2. **Logistic Regression for Batsman Classification**
- **Features**: Batting Average, Strike Rate
- **Target**: High-Performance Category (Ave >= 30)
- **Results**:
    - Model achieved **accuracy of ~69%**.
    - Identified top-performing batsmen.

#### 3. **Decision Tree Classifier for Bowling Performance Prediction**
- **Features**: Economy, Average, Strike Rate
- **Target**: High Wicket Taker (above median wickets)
- **Results**:
    - Decision tree model trained and evaluated.
    - Visualized the decision tree structure.
    - **Precision: 71%**, **Recall: 61%**, **F1-score: 68%**

#### 4. **Predicting Top 6 Bowlers**
- Used Decision Tree Classifier probabilities to rank bowlers.
- Identified **Top 6 Bowlers** based on model predictions.

### Unsupervised Learning Algorithm
#### **K-Means Clustering for Player Segmentation**
- **Features**: Runs, Balls Faced, 6s, Strike Rate
- **Clusters**: 4
- **Results**:
    - Clustered players based on performance metrics.
    - **Pairplot** was used to visualize the clusters and analyze similarities between players.
- **Performance Metrics**:
    - **Inertia**: Measures within-cluster variance.
    - **Silhouette Score**: **~0.54**, indicating well-defined clusters.

## Conclusion
- **Linear Regression** successfully predicted batting averages with good accuracy.
- **Logistic Regression** classified high-performing batsmen with ~69% accuracy.
- **Decision Tree Classifier** effectively predicted high wicket-takers.
- **K-Means Clustering** segmented players into meaningful groups.

## Future Improvements
- Use **Random Forest** for improved classification accuracy.
- Implement **Neural Networks** for more advanced predictions.
- Include **contextual game factors** (e.g., match situations) for enhanced player insights.

## How to Run the Project
1. Install dependencies:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```
2. Place the datasets (`t20.csv` and `Bowling_t20.csv`) in the project directory.
3. Run the Python script:
    ```bash
    python T20.py
    ```



