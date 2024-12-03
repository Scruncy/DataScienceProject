import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import ast
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scipy.stats import poisson
import plotly.graph_objects as go
import geopandas as gpd

# Same data manipulation done before
df = pd.read_pickle("dataset.pkl")
df = df[df['expo'] >= 0.1]
df['nclaims'] = np.round(np.divide(df['nclaims'], df['expo']))
df = df[df['nclaims'] < 6]
df = df.drop(columns=['expo'])
df['average'] = pd.to_numeric(df['average'], errors='coerce').fillna(0)
df['province'] = df['postcode'].astype(str).str[0].astype(int)
df = df.drop(columns=['postcode', 'id'])

def plot_barplot(arr, title, first, second):
    # Count occurrences of each unique value in the array
    unique_values, counts = np.unique(arr, return_counts=True)

    # Create a DataFrame for making life easier
    data = {'Value': unique_values, 'Count': counts}

    sns.barplot(x='Value', y='Count', data=pd.DataFrame(data))

    plt.xlabel(first)
    plt.ylabel(second)
    plt.title(title)

    plt.show()

def plot_categorical_histogram(arr, classes, title="", xlabel="", ylabel="Absolute Frequency"):

    # Categorize the data based on the custom class intervals that we take from the other file
    categorized_data = []
    for value in arr:
        for i in range(len(classes) - 1):
            if classes[i] <= value < classes[i + 1]:
                categorized_data.append(f'{classes[i]}-{classes[i + 1]}')
                break

    # to count how many values in each class
    counts = pd.Series(categorized_data).value_counts().sort_index()

    # Sort the counts by the numeric values of the class intervals
    counts.index = pd.Categorical(counts.index, categories=[f'{classes[i]}-{classes[i + 1]}' for i in range(len(classes) - 1)], ordered=True)
    counts = counts.sort_index()

    # Plot the histogram (bar chart)
    plt.bar(counts.index, counts.values, edgecolor='black')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(rotation=45)
    plt.show()

plot_barplot(df['claim'].values, 'Claim occured', '', 'Absolute frequency')
plot_barplot(df['province'].values, 'Provinces', 'Province', 'Absolute frequency')
plot_barplot(df['sex'].values, 'Gender', '', 'Absolute frequency')
plot_barplot(df['bm'].values, 'Bonus_malus classes', 'classes', 'Absolute frequency')
plot_barplot(df['coverage'].values, 'Type of coverage', '', 'Absolute frequency')
plot_categorical_histogram(df['power'], [0, 25, 50, 75, 100, 125, 150, 175, 200, 250], 'Power')
plot_categorical_histogram(df['ageph'], [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 150], 'Age of the policy-holder')
plot_categorical_histogram(df['agec'], [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], "Age of the vehicle")

# Plotting the data, compared with a real poisson with lambda the mean
empirical_data = df['nclaims'].values
lambda_ = np.mean(empirical_data)
# Generate values for the Poisson distribution
x_vals = np.arange(0, max(empirical_data) + 1)
poisson_probs = poisson.pmf(x_vals, lambda_)

# scale it to get the aboslute frequency
scaled_poisson = poisson_probs * len(empirical_data)

# histogram of empirical data
plt.figure(figsize=(8, 5))
plt.hist(
    empirical_data,
    bins=np.arange(min(empirical_data), max(empirical_data) + 2) - 0.5,
    color='#3498db',  # A calming blue
    alpha=0.7,
    label='Empirical Data Histogram',
    edgecolor='black'
)
# poisson distribution
plt.bar(
    x_vals,
    scaled_poisson,
    color='#e74c3c',  # A vibrant red
    alpha=0.6,
    width=1,
    label='Poisson Distribution',
    edgecolor='black'
)
# Added labels, legend, and title
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Empirical Data vs. Poisson Distribution', fontsize=14)
plt.legend(frameon=True, loc='upper right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()

#done the same as done in model_implementation
X_train, X_test, y_train, y_test = train_test_split(df, df['nclaims'], test_size=0.2, random_state=30)
bm = X_test['bm'].values
nclaims = y_test.values


# Plotting correlation matrix (Pearson and Kendall-Tau)
df_numerical = df[['nclaims', 'ageph', 'bm', 'power', 'agec', 'province']]
correlation_matrix = df_numerical.corr(method='pearson')
# Creating a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
# Display the plot
plt.title('Pearson Correlation Matrix')
plt.show()

#Same but for Kendall
correlation_matrix = df_numerical.corr(method='kendall')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Kendall Correlation Matrix')
plt.show()

#Hex bin plot for belgium
data = pd.DataFrame({
    'Longitude': df['long'],
    'Latitude': df['lat'],
    #'Value': df_test['nclaims']  # Optional: Weight values
})
#heatmap is another possibility, but we chose not to use since the other looks better
"""# Plot KDE heatmap
plt.figure(figsize=(8, 10))
sns.kdeplot(
    x=data['Longitude'], y=data['Latitude'],
    cmap='Reds', fill=True, bw_adjust=1
)
"""
# hexbin plot
plt.hexbin(data['Longitude'], data['Latitude'], gridsize=50, cmap='coolwarm', mincnt=1)

# Load Belgium map from downloaded shapefile (available online)
belgium_map = gpd.read_file("ne_10m_admin_0_sovereignty/ne_10m_admin_0_sovereignty.shp")
belgium_map[belgium_map['NAME'] == 'Belgium'].plot(ax=plt.gca(), edgecolor='black', facecolor='none')

plt.title("Hexbin Plot of Points in Belgium")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label='Density')
plt.show()

# same as other code
def compute_bm_class_array(current_classes, claims):

    current_classes = np.array(current_classes, dtype=int)
    claims = np.array(claims, dtype=int)
    new_classes = np.maximum(0, current_classes - 1)
    penalties = np.where((current_classes == 0) & (claims == 1), 4, claims * 5)
    new_classes = new_classes.astype(int)
    penalties = penalties.astype(int)
    new_classes += penalties
    new_classes = np.minimum(new_classes, 22)

    return new_classes

# Load the CSV containing the models
model_metadata_df = pd.read_csv("model_metadata.csv")
model_metadata_df['feature_names'] = model_metadata_df['feature_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# use the function to compute the predicted BM classes for each policy-holder in the test set
counter = 2 #to be changed to get the average of another model
predicted = np.array(eval(model_metadata_df['BM'].values[counter]), dtype=int)
real = compute_bm_class_array(bm, nclaims)

# Compute relative frequencies for "real" data
unique_real, counts_real = np.unique(real, return_counts=True)
relative_frequencies_real = counts_real / counts_real.sum()

# Create a Series for "real" data
real_series = pd.Series(relative_frequencies_real, index=unique_real, name='real')

# percentage of premia to be paid in each class
BM_premia = {
    22: 200,
    21: 160,
    20: 140,
    19: 130,
    18: 123,
    17: 117,
    16: 111,
    15: 105,
    14: 100,
    13: 95,
    12: 90,
    11: 85,
    10: 81,
    9: 77,
    8: 73,
    7: 69,
    6: 66,
    5: 63,
    4: 60,
    3: 57,
    2: 54,
    1: 54,
    0: 54
}

converted_data = [BM_premia[key] for key in predicted]
print(np.mean(converted_data))

converted_data = [BM_premia[key] for key in real]
print(np.mean(converted_data))  #avearge in real data set

# Initialize an empty DataFrame to store frequencies
frequency_df = pd.DataFrame()

# Loop through `i` from 0 to 7
for i in range(8):
    predicted = np.array(eval(model_metadata_df['BM'].values[i]), dtype=int)

    # Compute relative frequencies
    unique, counts = np.unique(predicted, return_counts=True)
    relative_frequencies = counts / counts.sum()
    # Create a Series for this iteration, with unique values as the index
    freq_series = pd.Series(relative_frequencies, index=unique, name=f'i_{i}')

    # Add this Series to the DataFrame
    frequency_df = pd.concat([frequency_df, freq_series], axis=1)

# Fill NaN values with 0 (in case some indices are missing for certain `i`)
frequency_df = pd.concat([frequency_df, real_series], axis=1)

frequency_df = frequency_df.fillna(0)

# Reset index for readability
frequency_df = frequency_df.reset_index().rename(columns={'index': 'Value'})

# Show the resulting DataFrame
print(frequency_df)




# Compare the arrays element-wise
equal_elements = np.equal(predicted, real)

# Count how many times the elements are the same (should be same result as sensitivity)
count_same = np.sum(equal_elements)

# Output the result
print(f"Number of matching elements: {count_same/len(predicted)}")

print(model_metadata_df)
print(model_metadata_df['feature_names'])

# Check data types
print(model_metadata_df.dtypes)
print(model_metadata_df['feature_importance'].head())  # Display the first few entries
print(model_metadata_df['feature_names'].head())  # Display the first few entries

## Extract and plot loss curves for each model
for index, row in model_metadata_df.iterrows():
    loss_values = row['loss_values']  # Directly access the loss values
    if isinstance(loss_values, str):  # If it's a string, convert it back to a list
        loss_values = eval(loss_values)
    if loss_values:  # Ensure loss_values is not None or empty
        plt.plot(loss_values, label=row['model_name'])

plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Extract feature importances
feature_importances = {}
for index, row in model_metadata_df.iterrows():
    feature_names = row['feature_names']  # List of feature names
    importance_values = eval(row['feature_importance'])  # Feature importances
    # Store importance values with feature names as columns
    feature_importances[row['model_name']] = dict(zip(feature_names, importance_values))

# Convert the dictionary to a DataFrame where each column is a model
importance_df = pd.DataFrame(feature_importances)

# Display the dataframe to ensure features are correctly saved
print(importance_df)

# Aggregate feature importances across models by summing them
total_importance = importance_df.sum(axis=1)

# Sort by total importance and get the top 10 features
top_10_features = total_importance.sort_values(ascending=False).head(10)

# Filter the importance_df to include only the top 10 features
top_10_importance_df = importance_df.loc[top_10_features.index]

# Plotting each model's feature importance for the top 10 features
top_10_importance_df.plot(kind='bar', figsize=(12, 6))

# Print the top 10 feature names to check
print("Top 10 Features:")
print(top_10_features.index)

# Set plot labels
plt.title('Feature Importance for Top 10 Features (Per Model)')
plt.xlabel('Features')
plt.ylabel('Total Importance')

# Use the correct feature names for the x-axis labels
plt.xticks(ticks=range(len(top_10_features)), labels=top_10_features.index, rotation=45, ha='right')

# Display the legend and plot
plt.legend(title="Models")
plt.tight_layout()
plt.show()

# Aggregate feature importances across models by averaging them
average_importance = importance_df.mean(axis=1)

# Sort by average importance and get the top 10 features with the highest average
top_10_average_features = average_importance.sort_values(ascending=False).head(10)

# Filter the importance_df to include only the top 10 features based on average importance
top_10_average_importance_df = importance_df.loc[top_10_average_features.index]

# Plotting each model's feature importance for the top 10 features with the highest average importance
top_10_average_importance_df.plot(kind='bar', figsize=(12, 6))

plt.title('Feature Importance for Top 10 Features with Highest Average Importance (Per Model)')
plt.xlabel('Features')
plt.ylabel('Average Importance')
plt.xticks(ticks=range(len(top_10_average_features)), labels=top_10_average_features.index, rotation=45, ha='right')
plt.legend(title="Models")
plt.tight_layout()
plt.show()



# Extract metrics for comparison
metrics = ['accuracy', 'precision', 'recall', 'f1']
metrics_df = model_metadata_df[['model_name'] + metrics]

# Melt the DataFrame for easier plotting
melted_df = metrics_df.melt(id_vars=['model_name'], var_name='Metric', value_name='Score')

# Plot using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(data=melted_df, x='Metric', y='Score', hue='model_name')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Function to clean and parse confusion matrix strings
def clean_confusion_matrix_string(confusion_matrix_str):
    if not isinstance(confusion_matrix_str, str):
        return confusion_matrix_str  # If already a valid object, return as-is

    # Remove extra spaces and commas
    confusion_matrix_str = confusion_matrix_str.replace(" ", ",").replace(",,", ",")
    # Remove any trailing commas in rows
    confusion_matrix_str = confusion_matrix_str.strip("[]").split("],")
    cleaned_rows = []
    for row in confusion_matrix_str:
        # Convert each row to a list of integers, replacing empty fields with 0
        cleaned_row = [int(num) if num else 0 for num in row.replace("[", "").replace("]", "").split(",")]
        cleaned_rows.append(cleaned_row)
    return np.array(cleaned_rows)

# Plot confusion matrices
for index, row in model_metadata_df.iterrows():
    # Convert list back to NumPy array
    confusion_matrix = np.array(eval(row['confusionM']))

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {row['model_name']}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


# Select a single feature and compare it against accuracy
feature_of_interest = "Feature 1"
correlation_data = {
    "Model": model_metadata_df['model_name'],
    "Feature Importance": [eval(row['feature_importance'])[0] for _, row in model_metadata_df.iterrows()],
    "Accuracy": model_metadata_df['accuracy']
}

correlation_df = pd.DataFrame(correlation_data)

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=correlation_df, x='Feature Importance', y='Accuracy', hue='Model', style='Model')
plt.title(f'Feature Importance vs. Accuracy ({feature_of_interest})')
plt.xlabel('Feature Importance')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

# Select performance metrics
heatmap_data = model_metadata_df.set_index('model_name')[['accuracy', 'precision', 'recall', 'f1']]

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Model Performance Metrics Heatmap')
plt.ylabel('Models')
plt.xlabel('Metrics')
plt.show()
