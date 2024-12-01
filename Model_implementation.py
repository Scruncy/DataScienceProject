###################################################
#################### LIBRARIES ####################
###################################################

# Data processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay

# Machine learning models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
)

# Statistical modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

# Deep learning
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Utility
from collections import Counter
import time

##############################################################################
# Create the function to compute the Bonus-Malus class for each policy-holder,
# knowing the current class (which referes to the class at the beginning of
# the policy) and the number of claims occured during the year


def compute_bm_class_array(current_classes, claims):

    # to be sure that both classes and claims are numpy arrays
    current_classes = np.array(current_classes, dtype=int)
    claims = np.array(claims, dtype=int)

    # Step 1: every year, the policy-holder gets a bonus of 1 class (if  not already in class 0), nevertheless the number of claims
    new_classes = np.maximum(0, current_classes - 1)

    # Step 2: for every claim, the policy-holder receives 5 malus, except if it has one claim and they were in class 0;
    # in that case the penalty for one claim is only 4
    penalties = np.where((current_classes == 0) & (claims == 1), 4, claims * 5)  # Special case for 0 class and 1 claim

    # convert into integer to be sure to deal with the correct type
    new_classes = new_classes.astype(int)
    penalties = penalties.astype(int)

    # apply penalties, but double check that the maximum class can be 22
    new_classes += penalties
    new_classes = np.minimum(new_classes, 22)  #can not exceed 22

    return new_classes


# the following code needs to be run only if dataset.pkl is not in the directory
# otherwise, it doesn't need to be run. This operation is due to get faster calculation time
"""
file_path = "beMTPL97.csv"
df = pd.read_csv(file_path)
df.to_pickle("dataset.pkl")
"""

# Load from pickle
df = pd.read_pickle("dataset.pkl")

# Data description:
# id: Numeric - Policy number.
# expo: Numeric - Exposure.
# claim: Factor - Indicates if a claim occurred.
# nclaims: Numeric - Number of claims.
# amount: Numeric - Aggregate claim amount.
# average: Numeric - Average claim amount.
# coverage: Factor - Insurance coverage level:
#   "TPL" = Third Party Liability only,
#   "TPL+" = TPL + Limited Material Damage,
#   "TPL++" = TPL + Comprehensive Material Damage.
# ageph: Numeric - Policyholder age.
# sex: Factor - Policyholder gender ("female" or "male").
# bm: Integer - Level occupied in the Belgian bonus-malus scale (0 to 22).
#       Higher levels indicate worse claim history (see Lemaire, 1995).
# power: Numeric - Horsepower of the vehicle in kilowatts.
# agec: Numeric - Age of the vehicle in years.
# fuel: Factor - Type of fuel of the vehicle ("gasoline" or "diesel").
# use: Factor - Use of the vehicle ("private" or "work").
# fleet: Integer - Indicates if the vehicle is part of a fleet (1 or 0).
# postcode: Postal code of the policyholder.
# long: Numeric - Longitude coordinate of the municipality where the policyholder resides.
# lat: Numeric - Latitude coordinate of the municipality where the policyholder resides.

print("DataFrame loaded:")
print(df.head())  # Display the first few rows of the DataFrame to understand what we deal with

#############################################################
####################### DATA CLEANING #######################
#############################################################

df['average'] = pd.to_numeric(df['average'], errors='coerce').fillna(0) # some data cleaning, namely fill with 0 where the value is "NaN"

#Let's check duplicates
duplicate_ids = df['id'][df['id'].duplicated()].unique()
print("Duplicate IDs:", duplicate_ids)
#No duplicates, really nice.
# dropping longitude and latitude since their value is represented by postcode
df.drop(columns=['long', 'lat'], inplace=True)
# just taking into consideration data where exposure is at least 0.1 (40 days more or less deep into the contract)
df = df[df['expo'] >= 0.1]
#divide number of claims by exposure to get the yearly value (assuming a poisson behaviour)
df['nclaims'] = np.round(np.divide(df['nclaims'], df['expo']))
# we remove outliers thast might be made after dividing number of claims by exposure
df = df[df['nclaims'] < 6]
# drop the exposure since not important anymore
df = df.drop(columns=['expo'])
#converties the postcode into the 9 belgian provinces (since all the postcode with the same first digit referes to the same province, only that value is taken into consideration)
df['province'] = df['postcode'].astype(str).str[0].astype(int)
df = df.drop(columns=['postcode'])

#for reproducibility of the code, seed 30 is chosen for everything (because it's a nice number)
# maybe check this part later
_, X_test, _, _ = train_test_split(df, df['nclaims'], test_size=0.2, random_state=30)
bm = X_test['bm'].values #current bm of the policyholder of the test vector

#getting dummies
df_dummies = pd.get_dummies(df, columns=['coverage', 'sex', 'fuel', 'use', 'fleet','province','bm'], drop_first=False)
print(df_dummies.head())
print(df_dummies.columns)

# remove all the features that are useless for our computation (or would have correlation 1 with the target variable)
X = df_dummies.drop(['claim', 'nclaims', 'amount', 'average', 'id', ], axis=1)  # Features
y = df_dummies['nclaims']               # Target variable

#let's find the numeric columns so we can scale them
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Apply MinMaxScaler to the numeric columns, to scale them
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Step 3: Split the dataset into training and testing sets (20% of the sample will be the test, random_state to be chosen for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

print(type(X_train), type(y_train))
print(X_train.dtypes, y_train.dtypes)

# Initialize an empty DataFrame for storing model metadata (which will be used in Visualization)
columns = ["model_name", "history", "mse", "mae", "accuracy", "precision", "recall", "f1", "confusionM",
           "loss_values", "predictions", "features", "feature_importance", "learning_rate", "epochs", "batch_size"]
model_metadata_df = pd.DataFrame(columns=columns)



#############################################
########### FUNCTION TO SAVE DATA ###########
#############################################

def collect_metadata(model_name, history, X_train, y_test, y_pred, newBM, learning_rate=None, training_duration=None):
    # Determine feature names
    if isinstance(X_train, np.ndarray):
        #giving some labels to features
        feature_names = [f"Feature {i + 1}" for i in range(X_train.shape[1])]
    else:
        feature_names = X_train.columns.tolist()

    # Initialize feature importance
    # needs of doing it because bagging and random don't save loss models in the same way
    if 'Bagging' in model_name or 'Random' in model_name:
        # Bagging and Random Forest: Use estimator's feature importance if available
        if hasattr(model, "estimators_"):
            # Average feature importances over all estimators
            feature_importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
        else:
            feature_importance = np.zeros(len(feature_names))
    elif 'NN' in model_name:
        # if it's a NN, all the datas are saved in the following way
        weights, biases = model.layers[0].get_weights()
        feature_importance = np.sum(np.abs(weights), axis=1)
        feature_importance = feature_importance / np.sum(feature_importance)  # Normalize
    else:
        # Default: Zero feature importance
        feature_importance = np.zeros(len(feature_names))

    # Initialize metadata dictionary to store values
    metadata = {
        "model_name": model_name,
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred) if len(set(y_test)) > 1 else None,
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=1) if len(set(y_test)) > 1 else None,
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=1) if len(set(y_test)) > 1 else None,
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=1) if len(set(y_test)) > 1 else None,
        "confusionM": confusion_matrix(y_test, y_pred).tolist() if len(set(y_test)) > 1 else None,
        "loss_values": history.history['loss'] if history and hasattr(history, 'history') else None,
        "predictions": y_pred.tolist(),
        "BM": newBM.tolist(),
        "feature_names": feature_names,
        "feature_importance": feature_importance.tolist(),
        "learning_rate": learning_rate,
        "epochs": len(history.epoch) if history and hasattr(history, 'epoch') else None,
        "train_loss": history.history.get('loss', []) if history and hasattr(history, 'history') else None,
        "val_loss": history.history.get('val_loss', []) if history and hasattr(history, 'history') else None,
        "train_accuracy": history.history.get('accuracy', []) if history and hasattr(history, 'history') else None,
        "val_accuracy": history.history.get('val_accuracy', []) if history and hasattr(history, 'history') else None,
        "model_architecture": [(layer.name, layer.get_config()) for layer in model.layers] if hasattr(model, 'layers') else None,
        "training_time": training_duration,
        "batch_size": 32 if history and hasattr(history, 'epoch') else None,
    }
    return metadata


# Our target variable y is Poisson distributed, so we hand build an adequate loss functions
# for the statistical learning algorithms: Poisson NLL. In the end this loss function has not been used
# but we left it since changing the code a little bit still allows to use this one to get slightly different
# results
def poisson_nll(y_true, y_pred):
    # The Poisson Negative Log-Likelihood
    # y_true is the ground truth (observed claims)
    # y_pred is the predicted rate (lambda), which should be positive

    # not to get log(0), epsilon really small is considered
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.maximum(y_pred, epsilon)  # Prevent negative or zero predictions

    # Poisson NLL formula: NLL = lambda - y * log(lambda) + log(y!)
    # Note: log(y!) is a constant so doesn't matter.
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred))

# Define the number of epochs (iterations) to run,
Nepochs = 10

#############################################################################################
# MODEL1: NN(15,25,10), with continuous variables
#############################################################################################


# Define the model with still continuous variables
model = Sequential([
    Dense(15, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation="softplus")
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=tf.keras.losses.Poisson())

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train,
          epochs=Nepochs,
          batch_size=32,
          verbose=1)

training_end_time = time.time()
training_duration = training_end_time - training_start_time

# Retrieve the learning rate
learning_rate = model.optimizer.learning_rate.numpy()

# Predict on the test set
# Classes [0, 0.3] --> 0, [0.3, 1.3] --> 1, [1.3,2.3] --> 2 and so on
def custom_mapping(predictions):
    return np.floor(predictions + 0.7).astype(int)

y_pred = custom_mapping(model.predict(X_test))
print(y_pred.flatten()) #to flatt
newBm = compute_bm_class_array(bm, y_pred.flatten())
# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="NN(15,20,10)Cont", history=history, X_train=X_train, y_test=y_test,
                                y_pred=y_pred, newBM = newBm, learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL2: NN(15,25,10), with categorical variables
#############################################################################################

#let's use again the df, but first some more data manipulation to increase the efficieny of the NN

# Converting some continuous features into categorical ones. (see book 3.4 for explanation why)
# Step 1: Define bins and labels for each feature

# Vehicle Age
vehicle_age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Bin edges for vehicle_age
vehicle_age_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '45+']  # Labels

# Policy Holder Age
policy_holder_age_bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 150]  # Bin edges for policy_holder_age
policy_holder_age_labels = ['18-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66+']  # Labels

# Vehicle Power
vehicle_power_bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]  # Bin edges for vehicle_power
vehicle_power_labels = ['0-25', '25-50', '50-75', '75-100', '100-125', '125-150', '150-175', '175-200', '200+']  # Labels

# Step 2: Apply pd.cut() to categorize each feature

df_dummies['vehicle_age_cat'] = pd.cut(df_dummies['agec'], bins=vehicle_age_bins, labels=vehicle_age_labels, right=False)
df_dummies['policy_holder_age_cat'] = pd.cut(df_dummies['ageph'], bins=policy_holder_age_bins, labels=policy_holder_age_labels, right=False)
df_dummies['vehicle_power_cat'] = pd.cut(df_dummies['power'], bins=vehicle_power_bins, labels=vehicle_power_labels, right=False)

# Drop the original continuous columns
df_dummies = df_dummies.drop(columns=['agec', 'ageph', 'power'])


# Step 3: Apply one-hot encoding to the newly created categorical columns

df_dummies = pd.get_dummies(df_dummies, columns=['vehicle_age_cat', 'policy_holder_age_cat', 'vehicle_power_cat'], prefix=['vehicle_age_cat', 'policy_holder_age_cat','vehicle_power_cat'], drop_first=False)

# Step 4: Check the new DataFrame
print(df_dummies.head())
print(df_dummies.columns)

# Step 2: Define features and target variable
X = df_dummies.drop(['claim', 'nclaims', 'amount', 'average', 'id'], axis=1)  # Features
y = df_dummies['nclaims']               # Target variable

# Preprocess numeric variables
# Identify the numeric columns for scaling (excluding the ones that are now categorical after pd.get_dummies())
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Print the shapes of the resulting datasets to verify
print("Training set (features):", X_train.shape)
print("Test set (features):", X_test.shape)
print("Training set (target):", y_train.shape)
print("Test set (target):", y_test.shape)

# Construct a Neural Network
model = Sequential([
    Dense(15, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dense(20, activation='relu'),  # Second hidden layer
    Dense(10, activation='relu'),  # Third hidden layer
    Dense(1, activation="softplus")  # Output layer for regression; softplus to ensure only pos output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Adam optimizer
              loss=tf.keras.losses.Poisson())  # Use Poisson NLL loss

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training
training_end_time = time.time()
training_duration = training_end_time - training_start_time
def custom_mapping(predictions):
    return np.floor(predictions + 0.7).astype(int)

y_pred = custom_mapping(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="NN(15,20,10)Cat", history=history, X_train=X_train, y_test=y_test,
                                y_pred=y_pred, newBM = compute_bm_class_array(bm, y_pred.flatten()),learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)


#############################################################################################
# MODEL3: NN(100,200,75), with categorical variables
#############################################################################################
# Construct a Neural Network
model = Sequential([
    Dense(100, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dense(200, activation='relu'),  # Second hidden layer
    Dense(75, activation='relu'),  # Third hidden layer
    Dense(1, activation="softplus")  # Output layer for regression; softplus to ensure only pos output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Adam optimizer
              loss=tf.keras.losses.Poisson())  # Use Poisson NLL loss

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training
training_end_time = time.time()
training_duration = training_end_time - training_start_time

def custom_mapping(predictions):
    return np.floor(predictions + 0.8).astype(int)

y_pred = custom_mapping(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="NN(100,200,75)Cat", history=history, X_train=X_train, y_test=y_test,
                                y_pred=y_pred, newBM = compute_bm_class_array(bm, y_pred.flatten()), learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL4: NN(500,1000,200), with categorical variables
#############################################################################################
# Construct a Neural Network
model = Sequential([
    Dense(500, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dense(1000, activation='relu'),  # Second hidden layer
    Dense(20, activation='relu'),  # Third hidden layer
    Dense(1, activation="softplus")  # Output layer for regression; softplus to ensure only pos output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Adam optimizer
              loss=tf.keras.losses.Poisson())  # Use Poisson NLL loss

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training
training_end_time = time.time()
training_duration = training_end_time - training_start_time
def custom_mapping(predictions):
    return np.floor(predictions + 0.7).astype(int)

y_pred = custom_mapping(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="NN(500,1000,200)Cat", history=history, X_train=X_train, y_test=y_test,
                                y_pred=y_pred, newBM = compute_bm_class_array(bm, y_pred.flatten()), learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL5: NN(72,200,100,5), with categorical variables
#############################################################################################
# Construct a Neural Network
model = Sequential([
    Dense(72, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dense(200, activation='relu'),  # Second hidden layer
    Dense(100, activation='relu'),  # Third hidden layer
    Dense(5, activation='relu'),  # Fourth hidden layer
    Dense(1, activation="softplus")  # Output layer for regression; softplus to ensure only pos output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Adam optimizer
              loss=tf.keras.losses.Poisson())  # Use Poisson NLL loss

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training
training_end_time = time.time()
training_duration = training_end_time - training_start_time
def custom_mapping(predictions):
    return np.floor(predictions + 0.7).astype(int)

y_pred = custom_mapping(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="NN(72,200,100,5)Cat", history=history, X_train=X_train, y_test=y_test,
                                y_pred=y_pred, newBM = compute_bm_class_array(bm, y_pred.flatten()), learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)


#############################################################################################
# MODEL6: GLM, with categorical variables
#############################################################################################
# Let's build a GLM-ish on the same training set and compare them
# Define the GLM-like neural network
model = Sequential([
    Dense(1, activation='exponential', input_dim=X_train.shape[1])  # Exponential link function
])

# Compile the model with Poisson loss
model.compile(optimizer='adam', loss='poisson')

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train, epochs=Nepochs, batch_size=32, verbose=1, validation_split=0.2)
training_end_time = time.time()
training_duration = training_end_time - training_start_time
# Predict using the trained Keras model
def custom_mapping(predictions):
    return np.floor(predictions + 0.7).astype(int)

y_pred = custom_mapping(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="GLMCat", history=history, X_train=X_train, y_test=y_test, y_pred=y_pred,
                                learning_rate=learning_rate, newBM = compute_bm_class_array(bm, y_pred.flatten()), training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL6: BaggingTree, with categorical variables
#############################################################################################

# Create a Bagging Classifier
model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=200,  # how many trees we want
        random_state=30,
        n_jobs=-1)

training_start_time = time.time()
# Train the model
history = model.fit(X_train, y_train)
training_end_time = time.time()
training_duration = training_end_time - training_start_time
# Predict using the trained Keras model
y_pred = np.round(model.predict(X_test).flatten())  # Flatten to convert predictions to 1D array

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="BaggingTreeCat", history=history, X_train=X_train, y_test=y_test, y_pred=y_pred,
                                learning_rate=learning_rate, newBM = compute_bm_class_array(bm, y_pred.flatten()), training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL7: Random Forest, with categorical variables
#############################################################################################


# Create a Random Forest Classifier
model = RandomForestClassifier(
        n_estimators=200,
        random_state=30,
        n_jobs=-1)

# Train the model
training_start_time = time.time()
# Train the model
history = model.fit(X_train, y_train)
training_end_time = time.time()
training_duration = training_end_time - training_start_time
# Predict using the trained Keras model
y_pred = np.round(model.predict(X_test).flatten())  # Flatten to convert predictions to 1D array

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="RandomForestCat", history=history, X_train=X_train, y_test=y_test, y_pred=y_pred,
                                learning_rate=learning_rate, newBM = compute_bm_class_array(bm, y_pred.flatten()), training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)


######################################################################
# Save the DataFrame to a CSV file, to work on it later on
model_metadata_df.to_csv("model_metadata.csv", index=False)
