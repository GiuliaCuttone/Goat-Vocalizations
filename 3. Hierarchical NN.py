"""
Goat vocalizations
Università degli studi di Milano

@author: Giulia Cuttone
"""

import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow.keras.layers as layers
import pandas as pd

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


'---------------------------------------------------------------------------------------------------'

# Early Stopping and Learning Rate Scheduler Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Class lists
emotional_labels = ["Negative", "Positive"]
class_labels = ["Calori", "Distribuzione Cibo", "Fenomeni legati al parto", "Ferita-Morte",
                "Isolamento sociale", "Presenza contemporanea di madri e capretti",
                "Separazione madre capretto", "Visita di estranei"]


# Set up parameter grids for Neural Network models:
param_grid_level1 = {
    'batch_size': [16, 32, 64],
    'epochs': [120, 200, 250] }

param_dist_level2 = {
    'batch_size': [8, 16, 32],
    'epochs': [120, 200, 250] }


'---------------------------------------------------------------------------------------------------'

"Neural Network Models:"

def Emotional_Classifier(input_shape, learning_rate=0.001, num_neurons=128, dropout_rate=0.3):
    
    """
    Builds and compiles a neural network model for binary classification.
    
    This model is designed to classify input data into two classes.
    It consists of three dense blocks with batch normalization and dropout layers for regularization,
    with a final sigmoid-activated output layer suitable for binary classification.
    -----------------------------------------------------------------------------------------------------
    Parameters:
        input_shape (int or tuple): Specifies the shape of the input data (number of features per sample).
        learning_rate (float, optional): Learning rate for the Adam optimizer; default is 0.001.
        num_neurons (int, optional): Number of neurons in the first and third dense layers.
            The second layer contains twice this number of neurons; default is 128.
        dropout_rate (float, optional): Dropout rate applied after each dense layer; default is 0.3.
    ----------------------------------------------------------------------------------------------------
    Returns:
        model (tf.keras.Sequential): A compiled TensorFlow Keras Sequential model ready for training.
    """
    
    # Define the model
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    # First Dense Block
    model.add(layers.Dense(num_neurons, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Second Dense Block with increased neurons
    model.add(layers.Dense(num_neurons * 2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Third Dense Block
    model.add(layers.Dense(num_neurons, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Output Layer
    model.add(layers.Dense(1, activation='sigmoid'))    # Sigmoid activation for binary classification
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model



def Class_Classifier(input_shape, learning_rate=0.001, num_neurons=256, dropout_rates=(0.4, 0.3, 0.2, 0.1)):   

    """
    Builds and compiles a neural network model for multi-class classification.
    
    This model is designed to classify input data into multiple classes,
    with a final softmax-activated output layer for multi-class prediction.
    The architecture incorporates progressively narrower dense layers to reduce model complexity as layers deepen,
    alongside adjustable dropout rates for each layer to prevent overfitting.
    -----------------------------------------------------------------------------------------------------
    Parameters:
        input_shape (int or tuple): Specifies the shape of the input data (number of features per sample).
        learning_rate (float, optional): Learning rate for the Adam optimizer; default is 0.001.
        num_neurons (int, optional): Number of neurons in the first (widest) dense layer.
            Each subsequent layer reduces the number of neurons by half; default is 256.
        dropout_rates (tuple of floats, optional): Dropout rates for each dense layer,
            where each element in the tuple specifies the dropout rate for the corresponding layer.
            The default values are set as (0.4, 0.3, 0.2, 0.1), reducing in later layers
            to maintain key features while preventing overfitting in earlier layers.
    ----------------------------------------------------------------------------------------------------
    Returns:
        model (tf.keras.Sequential): A compiled TensorFlow Keras Sequential model ready for training.
    """
    
    # Define the model
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    # First Dense Block
    model.add(layers.Dense(num_neurons, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rates[0]))
    
    # Second Dense Block with reduced neurons
    model.add(layers.Dense(num_neurons // 2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rates[1]))

    # Third Dense Block with further reduced neurons
    model.add(layers.Dense(num_neurons // 4, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rates[2]))
    
    # Fourth Dense Block with minimal neurons
    model.add(layers.Dense(num_neurons // 8, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rates[3]))    
    
    # Output Layer
    model.add(layers.Dense(4, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


'---------------------------------------------------------------------------------------------------'

def final_classification(X_test, y, predictions, clf_pos, clf_neg):
    
    """
    Final classification based on Level 1 predictions.
    
    Uses the Level 1 predictions ('Positive' or 'Negative')
    to determine which Level 2 classifier should be used for the final classification.
    If the Level 1 prediction is 'Positive', the function uses the `clf_pos` classifier.
    If the prediction is 'Negative', it uses the `clf_neg` classifier.
    If the label is neither 'Positive' nor 'Negative', the function returns 'Unknown'.
    ----------------------------------------------------------------------------------
    Paramenters:
        y (list): Labels from Level 1 classifier predictions ('Positive' or 'Negative').
        predictions (list): List to store the final predictions.
        clf_pos (classifier): Level 2 classifier for positive classes.
        clf_neg (classifier): Level 2 classifier for negative classes.
    --------------------------------------------------------------------
    Returns:
        predictions (list): A list of final predictions for each observation. 
                            Returns 'Unknown' for invalid labels.
    """
    
    for i, x_test in enumerate(X_test):
        if y[i] == 1:
            predictions.append(clf_pos.best_estimator_.predict([x_test])[0])
        else:
            predictions.append(clf_neg.best_estimator_.predict([x_test])[0])
    return predictions


"Plot function:"

def plot_confusion_matrix(cm, size, class_names, title='Confusion Matrix'):
    
    """
    Displays a confusion matrix as a heatmap.
    -----------------------------------------
    Parameters:
        cm (ndarray): Confusion matrix (2D array).
        size ((float, float)): Size of the confusion matrix.
        class_names (list or array): List of class labels to be displayed on the x and y axes.
        title (str): Title of the plot (default: 'Confusion Matrix').
    ------------------------------------------------------------------------------------------
    Returns:
        None: Displays the confusion matrix as a heatmap.
    """
    
    plt.figure(figsize=size)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


'---------------------------------------------------------------------------------------------------'

"Main code:"

# Load dataset
file_path = './Vocapra_postprocessing.csv'
df = pd.read_csv(file_path)

# Features (X) and labels (y) extraction
X = df.drop(columns=['Class', 'Emotional_state']).values   # Features
y1 = df['Emotional_state'].values                          # Emotional states
y2 = df['Class'].values                                    # Classes


# Encode the labels into numeric form
y1_encoded = np.array([emotional_labels.index(label) for label in y1])
y2_encoded = np.array([class_labels.index(label) for label in y2])

# Split the dataset into train and test sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1_encoded, y2_encoded, test_size=0.2, random_state=42)

input_shape = X_train.shape[1]


# Initialize the Neural Network classifier for Level 1
model1 = KerasClassifier(Emotional_Classifier, input_shape=input_shape, verbose=0)

# Train Level 1 classifier with grid search
grid_level1 = GridSearchCV(model1, param_grid_level1, cv=5, n_jobs=-1, verbose=1)
grid_level1.fit(X_train, y1_train, callbacks=[early_stopping, lr_scheduler])

# Make predictions
y1_pred = grid_level1.best_estimator_.predict(X_test)


# Decoding Level 1 Predictions and True Labels
y1_test_str = [emotional_labels[label] for label in y1_test]
y1_pred_str = [emotional_labels[label] for label in y1_pred]

# Model evaluation:
print("\nLevel 1 classification report:\n")

print("Best parameters:", grid_level1.best_params_, "\n")       # {'epochs': 250, 'batch_size': 16} 
print(classification_report(y1_test_str, y1_pred_str))          # Accuracy: 97%

# Confusion matrix:
cm_1 = confusion_matrix(y1_test_str, y1_pred_str, labels=emotional_labels)
plot_confusion_matrix(cm_1, (4, 3), emotional_labels, 'Level 1 Confusion Matrix')


'---------------------------------------------------------------------------------------------------'

'Level 2: Final Classification (Based on Level 1 prediction, train separate classifiers for Level 2)'

# Split data for Level 2 based on predicted emotional states
X_train_neg = X_train[y1_train == 0]
y2_train_neg = y2_train[y1_train == 0]

X_train_pos = X_train[y1_train == 1]
y2_train_pos = y2_train[y1_train == 1]


# Initialize the Neural Network Classifiers for Level 2 (Positive & Negative)
model2_pos = KerasClassifier(Class_Classifier, input_shape=input_shape, verbose=0)
model2_neg = KerasClassifier(Class_Classifier, input_shape=input_shape, verbose=0)

# Train Level 2 classifiers with Randomized Search
random_search_pos = RandomizedSearchCV(model2_pos, param_dist_level2, n_iter=9, cv=5, random_state=42, n_jobs=-1, verbose=1)
random_search_pos.fit(X_train_pos, y2_train_pos, callbacks=[early_stopping, lr_scheduler])

random_search_neg = RandomizedSearchCV(model2_neg, param_dist_level2, n_iter=9, cv=5, random_state=42, n_jobs=-1, verbose=1)
random_search_neg.fit(X_train_neg, y2_train_neg, callbacks=[early_stopping, lr_scheduler])


# Final predictions (hierarchical)
final_predictions = []
final_classification(X_test, y1_pred, final_predictions, random_search_pos, random_search_neg)


# Decoding Level 2 Predictions and True Labels
y2_test_str = [class_labels[label] for label in y2_test]
final_predictions_str = [class_labels[label] for label in final_predictions]

# Models evaluation:
print("\nLevel 2 classification report:\n")

print("Best parameters (Positive):", random_search_pos.best_params_)           # {'epochs': 200, 'batch_size': 16} 
print("Best parameters (Negative):", random_search_neg.best_params_, "\n")     # {'epochs': 200, 'batch_size': 32} 
print(classification_report(y2_test_str, final_predictions_str))               # Accuracy: 88%

# Confusion matrix:
cm_2 = confusion_matrix(y2_test_str, final_predictions_str, labels=class_labels)
plot_confusion_matrix(cm_2, (10, 7), class_labels, 'Level 2 Confusion Matrix')

'-----------------------------------------------------------------------------------'

# Save the models
#grid_level1.best_estimator_.model.save('level1_emotional_model.h5')
#random_search_pos.best_estimator_.model.save('level2_positive_model.h5')
#random_search_neg.best_estimator_.model.save('level2_negative_model.h5')
