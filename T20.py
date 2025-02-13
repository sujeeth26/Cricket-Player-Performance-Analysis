#!/usr/bin/env python
# coding: utf-8

# # Loading the datasets, Performing EDA and training the dataset using supervised learning algorithms

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the datasets
bowling_data = pd.read_csv('Bowling_t20.csv')
batting_data = pd.read_csv('t20.csv')

# Cleaning and preparing the batting data
batting_data.drop(['Unnamed: 0', 'Span', 'Mat', 'Inns'], axis=1, inplace=True, errors='ignore')
batting_data['Runs'] = pd.to_numeric(batting_data['Runs'].str.replace('*', ''), errors='coerce')
batting_data['BF'] = pd.to_numeric(batting_data['BF'], errors='coerce')
batting_data['SR'] = pd.to_numeric(batting_data['SR'], errors='coerce')
batting_data['Ave'] = pd.to_numeric(batting_data['Ave'], errors='coerce')

# Fill NaNs with the mean of the column
for column in batting_data.columns:
    if batting_data[column].dtype == 'float64' or batting_data[column].dtype == 'int64':
        batting_data[column].fillna(batting_data[column].mean(), inplace=True)

# Define features and target for batsmen
X_bat = batting_data[['Runs', 'BF', 'SR']]
y_bat = batting_data['Ave']

# Split the data into training and testing sets for batsmen
X_train_bat, X_test_bat, y_train_bat, y_test_bat = train_test_split(X_bat, y_bat, test_size=0.2, random_state=42)

# Linear Regression for batsmen
lr_bat = LinearRegression()
lr_bat.fit(X_train_bat, y_train_bat)
batting_data['predicted_ave'] = lr_bat.predict(X_bat)

# Sort batsmen based on predicted averages and their actual strike rates
top_batsmen = batting_data.sort_values(by=['predicted_ave', 'SR'], ascending=[False, False]).head(10)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(10, 12))
top_batsmen.plot(kind='bar', x='Player', y='predicted_ave', ax=ax[0], color='skyblue')
ax[0].set_title('Top 10 Batsmen by Predicted Average')
ax[0].set_ylabel('Predicted Average')

top_batsmen.plot(kind='bar', x='Player', y='SR', ax=ax[1], color='lightgreen')
ax[1].set_title('Top 10 Batsmen by Strike Rate')
ax[1].set_ylabel('Strike Rate')

plt.tight_layout()
plt.show()


# # Feature Engineering Methods

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the batting data
batting_data_path = 't20.csv'
batting_data = pd.read_csv(batting_data_path)
batting_data['Ave'] = pd.to_numeric(batting_data['Ave'], errors='coerce')
batting_data['Inns'] = pd.to_numeric(batting_data['Inns'], errors='coerce')
batting_data.dropna(subset=['Ave', 'Inns'], inplace=True)
batting_data['Weighted_Ave'] = batting_data['Ave'] * batting_data['Inns']

# Load the bowling data
bowling_data_path = 'Bowling_t20.csv'
bowling_data = pd.read_csv(bowling_data_path)
bowling_data['Wkts'] = pd.to_numeric(bowling_data['Wkts'], errors='coerce')
bowling_data['Mat'] = pd.to_numeric(bowling_data['Mat'], errors='coerce')
bowling_data.dropna(subset=['Wkts', 'Mat'], inplace=True)
bowling_data['Wkts_per_Match'] = bowling_data['Wkts'] / bowling_data['Mat']

# Sort and select top 10 for visualization
top_10_batsmen = batting_data.sort_values('Weighted_Ave', ascending=False).head(10)
top_10_bowlers = bowling_data.sort_values('Wkts_per_Match', ascending=False).head(10)

# Plotting top 10 batsmen by Weighted Average
plt.figure(figsize=(12, 6))
sns.barplot(x='Weighted_Ave', y='Player', data=top_10_batsmen, palette='coolwarm')
plt.title('Top 10 Batsmen by Weighted Average')
plt.xlabel('Weighted Average')
plt.ylabel('Player')
plt.show()

# Plotting top 10 bowlers by Wickets per Match
plt.figure(figsize=(12, 6))
sns.barplot(x='Wkts_per_Match', y='Player', data=top_10_bowlers, palette='viridis')
plt.title('Top 10 Bowlers by Wickets per Match')
plt.xlabel('Wickets per Match')
plt.ylabel('Player')
plt.show()


# # Implementing Logistic Algorithm

# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the batting data
batting_data_path = 't20.csv'  # Make sure to replace 'path_to_your_data/t20.csv' with the actual path
batting_data = pd.read_csv(batting_data_path)

# Convert 'Ave' and 'SR' columns to numeric, removing non-numeric entries and missing values
batting_data['Ave'] = pd.to_numeric(batting_data['Ave'], errors='coerce')
batting_data['SR'] = pd.to_numeric(batting_data['SR'], errors='coerce')
batting_data.dropna(subset=['Ave', 'SR'], inplace=True)

# Define a binary target based on batting average, assuming high performance as average 30 or more
batting_data['High_Performance'] = (batting_data['Ave'] >= 30).astype(int)

# Features and target
X = batting_data[['Ave', 'SR']]
y = batting_data['High_Performance']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Logistic Regression
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)

# Predicting on test set
log_predictions = log_model.predict(X_test_scaled)
log_accuracy = accuracy_score(y_test, log_predictions)

# Print model performances
print(f"Logistic Regression Accuracy: {log_accuracy}")

# Sort the data by Strike Rate in descending order
sorted_batting_data = batting_data.sort_values(by='SR', ascending=False)

# Select the top 10 batsmen with the highest strike rate
top_10_batsmen = sorted_batting_data[['Player', 'SR']].head(10)

print(top_10_batsmen)

# Calculate and print the accuracy of the model in percentage format
log_accuracy_percentage = accuracy_score(y_test, log_predictions) * 100
print(f"The accuracy of the logistic regression model is: {log_accuracy_percentage:.2f}%")



# # Implementing supervised learning Algorithm- Decision Tree

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Bowling_t20 (1).csv')

# Convert columns to numeric, explicitly handling any conversion issues
data['Econ'] = pd.to_numeric(data['Econ'], errors='coerce')
data['Ave'] = pd.to_numeric(data['Ave'], errors='coerce')
data['SR'] = pd.to_numeric(data['SR'], errors='coerce')
data['Wkts'] = pd.to_numeric(data['Wkts'], errors='coerce')  # Ensure 'Wkts' is numeric

# Drop any rows with NaN values created by unsuccessful conversions
data.dropna(subset=['Econ', 'Ave', 'SR', 'Wkts'], inplace=True)

# Define the target variable: high wicket taker (1 if above median wickets, 0 otherwise)
median_wickets = data['Wkts'].median()
data['High Wicket Taker'] = (data['Wkts'] > median_wickets).astype(int)

# Select features
features = data[['Econ', 'Ave', 'SR']]
target = data['High Wicket Taker']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test set
predictions = classifier.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Visualize the tree
plt.figure(figsize=(20,10))
tree.plot_tree(classifier, filled=True, feature_names=['Econ', 'Ave', 'SR'], class_names=['Low', 'High'])
plt.show()


# In[ ]:


#Accuracy: The model accurately predicts outcomes 69% of the time.
Precision: For positive predictions, the model is correct about 71% of the time.
Recall: It correctly identifies 61% of all actual positive cases.
F1-Score: The model achieves an F1-score of approximately 0.68, indicating a good balance between precision and recall.

Overall, the model performs well in identifying and predicting outcomes with a solid balance of accuracy and detail in decision-making, as visualized in the extensive decision tree structure


# # Predicting Top 6 Bowlers

# In[26]:


import pandas as pd
# Calculate the probability of each bowler being a high wicket taker
# classifier.predict_proba gives the probabilities for all classes; the second column ([,1]) gives the probability of being a high wicket taker
probabilities = classifier.predict_proba(data[['Econ', 'Ave', 'SR']])[:, 1]

# Add these probabilities back to the original DataFrame
data['High_Wicket_Taker_Probability'] = probabilities

# Sort the bowlers by their probability of being a high wicket taker
top_bowlers = data.sort_values(by='High_Wicket_Taker_Probability', ascending=False)

# Select the top 6 bowlers
top_6_bowlers = top_bowlers.head(6)

# Display the top 6 bowlers
print(top_6_bowlers[['Player', 'High_Wicket_Taker_Probability']])


# # Implementing Unsupervised Learning Algorithm - K Means Clustering

# In[2]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('t20.csv')

# Convert columns to numeric, check and handle if any conversions fail
try:
    data['Runs'] = pd.to_numeric(data['Runs'], errors='coerce')
    data['BF'] = pd.to_numeric(data['BF'], errors='coerce')
    data['6s'] = pd.to_numeric(data['6s'], errors='coerce')
    data['SR'] = pd.to_numeric(data['SR'], errors='coerce')
except Exception as e:
    print("Error in converting to numeric:", e)

# Drop any rows with NaN values created by unsuccessful conversions
data.dropna(subset=['Runs', 'BF', '6s', 'SR'], inplace=True)

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data[['Runs', 'BF', '6s', 'SR']])

# Applying K-means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(features_scaled)

# Add cluster information back to the original data
data['Cluster'] = clusters

# Visualization of clusters using pairplot
sns.pairplot(data, hue='Cluster', vars=['Runs', 'BF', '6s', 'SR'], palette='viridis')
plt.title('Clusters of Players Based on Performance Metrics')
plt.show()

# Analyze the centroids to understand the typical characteristics of each cluster
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['Runs', 'BF', '6s', 'SR'])
print(centroids)


# # Performance of K means 

# In[24]:


from sklearn.metrics import silhouette_score

# Calculate Inertia: Sum of squared distances of samples to their closest cluster center
inertia = kmeans.inertia_
print(f'Inertia: {inertia}')

# Calculate Silhouette Score: Mean Silhouette Coefficient for all samples
silhouette_avg = silhouette_score(features_scaled, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the top players based on 'predicted_ave'
plt.figure(figsize=(10, 5))
sns.barplot(x='Player', y='predicted_ave', data=top_batsmen, palette='coolwarm')
plt.title('Top 9 Batsmen Based on Predicted Average')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




