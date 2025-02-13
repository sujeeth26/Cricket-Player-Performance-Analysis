Predicting T20 Cricket player Performance Using Machine Learning

Overview

This project utilizes supervised and unsupervised learning algorithms to analyze cricket batting and bowling performances using T20 cricket data. It includes:

Exploratory Data Analysis (EDA)

Feature Engineering

Supervised Learning (Linear Regression, Logistic Regression, Decision Tree Classifier)

Unsupervised Learning (K-Means Clustering)

Performance Evaluation and Visualization

Dataset

The project uses two datasets:

t20.csv - Contains batting statistics (Runs, Balls Faced, Strike Rate, etc.)

Bowling_t20.csv - Contains bowling statistics (Wickets, Economy Rate, etc.)

Methodology

1. Data Preprocessing & EDA

Data is cleaned and missing values are handled.

Numeric conversions are performed where necessary.

Feature selection is done for relevant variables.

Exploratory Data Analysis (EDA) visualizes distributions, trends, and correlations.

2. Supervised Learning Algorithms

Linear Regression (For Batting Performance Prediction)

Predicts batting average based on Runs, Balls Faced, and Strike Rate.

Top batsmen are identified based on predicted averages and strike rates.

Visualization: Bar charts for top batsmen by predicted average and strike rate.

Logistic Regression (Classifying High-Performance Batsmen)

Defines high-performing batsmen as those with an average >= 30.

Accuracy: Evaluated using test data.

Top 10 batsmen by strike rate displayed.

Decision Tree Classifier (For Bowling Performance)

Predicts whether a bowler is a high wicket-taker based on economy rate, average, and strike rate.

Confusion matrix & classification report generated for performance evaluation.

Decision tree visualization helps in understanding model decisions.

Top 6 bowlers identified based on their probability of being a high wicket-taker.

3. Unsupervised Learning Algorithm

K-Means Clustering (Grouping Players Based on Performance)

Players are clustered based on Runs, Balls Faced, Sixes, and Strike Rate.

Optimal clusters identified using inertia and silhouette score.

Pair plots used for cluster visualization.

Cluster centroids analyzed to identify player groups.

Results & Performance Evaluation

Linear Regression Results

The model predicts batting averages effectively.

Top 10 batsmen by predicted average and strike rate are displayed in a bar chart.

Logistic Regression Results

Achieves good accuracy in classifying high-performance batsmen.

The accuracy score is displayed in percentage format.

Top batsmen by strike rate are shown in tabular format.

Decision Tree Classifier Results

The model identifies high wicket-taking bowlers with good precision.

Confusion matrix & classification report provide insights into model performance.

A decision tree plot visualizes model decisions.

Top 6 bowlers based on probability are listed.

K-Means Clustering Results

The model successfully groups players into 4 performance-based clusters.

Silhouette Score evaluates clustering effectiveness.

Pair plots visualize player clusters.

Cluster centroids provide insights into typical player performance in each cluster.

Visualizations

Bar charts for top batsmen by predicted average and strike rate.

Bar chart for top 6 bowlers by high wicket-taking probability.

Bar chart for silhouette score evaluation in K-Means clustering.

Decision tree plot for bowler classification.

Confusion matrix and classification report for decision tree performance.

Pair plot for K-Means clusters.

Conclusion

Linear Regression effectively predicts batting averages.

Logistic Regression classifies high-performing batsmen with good accuracy.

Decision Tree Classifier successfully identifies top-performing bowlers.

K-Means Clustering groups players into meaningful clusters.

Feature Engineering improves model performance.

The project provides data-driven insights into cricket performance, valuable for selectors, analysts, and fantasy leagues.

Future Scope

Integrate more advanced ML models (Random Forest, XGBoost, Neural Networks).

Incorporate real-time cricket data streaming for dynamic analysis.

Build a Streamlit Web App for interactive visualizations and predictions.

