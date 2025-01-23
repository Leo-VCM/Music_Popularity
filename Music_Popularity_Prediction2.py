import pandas as pd  # Import pandas for data manipulation

# Load the Spotify data from a CSV file into a DataFrame
spotify_data = pd.read_csv("Spotify_data.csv")

# Display the first few rows of the dataset to understand its structure
print(spotify_data.head())

# Drop the unnamed column (usually an index column from the CSV)
spotify_data.drop(columns=['Unnamed: 0'], inplace=True)

# Display general information about the DataFrame such as column types and non-null counts
spotify_data.info()

import matplotlib.pyplot as plt  # Import matplotlib for plotting
import seaborn as sns  # Import seaborn for statistical plotting

# Define the features of interest for analysis
features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness']

# Loop through the list of features and plot a scatterplot for each feature vs. Popularity
for feature in features:
    plt.figure(figsize=(8, 5))  # Set figure size for each plot
    sns.scatterplot(data=spotify_data, x=feature, y='Popularity')  # Create scatterplot
    plt.title(f'Popularity vs {feature}')  # Set plot title
    plt.show()  # Display the plot

# Select only the numeric columns from the DataFrame
numeric_columns = spotify_data.select_dtypes(include=['float64', 'int64']).columns
numeric_data = spotify_data[numeric_columns]

# Compute the correlation matrix for numeric columns
corr_matrix = numeric_data.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')  # Set plot title
plt.show()  # Display the heatmap

# Plot the distribution of each feature
for feature in features:
    plt.figure(figsize=(8, 5))  # Set figure size
    sns.histplot(spotify_data[feature], kde=True)  # Create histogram with a KDE curve
    plt.title(f'Distribution of {feature}')  # Set plot title
    plt.show()  # Display the plot

# Import necessary machine learning libraries
from sklearn.model_selection import train_test_split  # Function for splitting data
from sklearn.model_selection import GridSearchCV  # Function for performing grid search for hyperparameter tuning
from sklearn.ensemble import RandomForestRegressor  # Random Forest model
from sklearn.preprocessing import StandardScaler  # Scaler for normalizing features
from sklearn.metrics import mean_squared_error, r2_score  # Metrics for evaluating model performance

# Select the features (input variables) and the target variable (Popularity)
features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness', 'Tempo', 'Speechiness', 'Liveness']
X = spotify_data[features]  # Store the selected features in X
y = spotify_data['Popularity']  # Store the target variable (Popularity) in y

# Split the dataset into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data using StandardScaler (to improve model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)  # Transform the test data

# Define the hyperparameter grid for tuning the Random Forest model
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for each split
    'max_depth': [10, 20, 30, None],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Perform grid search to find the best hyperparameters for the Random Forest model
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, refit=True, verbose=2, cv=5)

# Fit the grid search to the training data
grid_search_rf.fit(X_train_scaled, y_train)

# Get the best hyperparameters found during grid search
best_params_rf = grid_search_rf.best_params_

# Retrieve the best Random Forest model using the best hyperparameters
best_rf_model = grid_search_rf.best_estimator_

# Use the best model to predict on the test data
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

# Create a scatter plot comparing actual vs. predicted popularity values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best_rf, alpha=0.7)  # Plot actual vs predicted values
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # Add a red line for perfect predictions
plt.xlabel('Actual Popularity')  # Label for the x-axis
plt.ylabel('Predicted Popularity')  # Label for the y-axis
plt.title('Actual vs Predicted Popularity (Best Random Forest Model)')  # Title of the plot
plt.show()  # Display the plot
