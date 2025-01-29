"""
Goat vocalizations
Università degli studi di Milano

@author: Giulia Cuttone
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


'---------------------------------------------------------------------------------------------------'

def final_classification(y, predictions, clf_pos, clf_neg):
    
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
        if y[i] == 'Positive':
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

# Split data into training and test set
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.2, random_state=42)


# Set up parameter grids for Random Forest:
param_grid = {
    'n_estimators': [100, 200, 300],            # Number of trees
    'max_depth': [20, 30, 50, None]             # Maximum depth of each tree
    }

param_dist = {
    'n_estimators': [300, 500, 800],  # Number of trees
    'max_depth': [20, 30, 50, None],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10]   # Minimum number of samples to split a node
    }

'---------------------------------------------------------------------------------------------------'

'Level 1: Positive vs Negative states'

# Random Forest Classifier
rf1 = RandomForestClassifier()
grid_search = GridSearchCV(rf1, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y1_train)
y1_pred_rf = grid_search.best_estimator_.predict(X_test)


# Model evaluation:
print("\nLevel 1 classification report:\n")
    
print("Random Forest Classifier:")
print("Best parameters:", grid_search.best_params_, "\n")       # {'max_depth': 30, 'n_estimators': 300}  
print(classification_report(y1_test, y1_pred_rf))               # Accuracy: 91%

# Confusion matrix:
cm_1 = confusion_matrix(y1_test, y1_pred_rf, labels=grid_search.classes_)
plot_confusion_matrix(cm_1, (4, 3), grid_search.classes_, 'Level 1 Confusion Matrix')


'---------------------------------------------------------------------------------------------------'

'Level 2: Final Classification (Based on Level 1 prediction, train separate classifiers for Level 2)'

# Split the data based on predicted labels
X_train_pos = X_train[y1_train == 'Positive']
y2_train_pos = y2_train[y1_train == 'Positive']

X_train_neg = X_train[y1_train == 'Negative']
y2_train_neg = y2_train[y1_train == 'Negative']


# Initialize the Random Forest Classifiers for Level 2 (Positive & Negative)
rf2_pos = RandomForestClassifier()
rf2_neg = RandomForestClassifier()

# Train Level 2 classifiers with Randomized Search
random_search_pos = RandomizedSearchCV(rf2_pos, param_dist, n_iter=20, cv=5, random_state=42)
random_search_pos.fit(X_train_pos, y2_train_pos)

random_search_neg = RandomizedSearchCV(rf2_neg, param_dist, n_iter=20, cv=5, random_state=42)
random_search_neg.fit(X_train_neg, y2_train_neg)

# Final predictions (hierarchical)
final_predictions_rf = []
final_classification(y1_pred_rf, final_predictions_rf, random_search_pos, random_search_neg)


# Models evaluation:
print("\nLevel 2 classification report:\n")
    
print("Random Forest Classifier:")
print("Best parameters (Positive):", random_search_pos.best_params_)            # {'n_estimators': 500, 'min_samples_split': 5, 'max_depth': 30}
print("Best parameters (Negative):", random_search_neg.best_params_, "\n")      # {'n_estimators': 800, 'min_samples_split': 2, 'max_depth': None} 
print(classification_report(y2_test, final_predictions_rf))                     # Accuracy: 75%

# Confusion matrix:
cm_2 = confusion_matrix(y2_test, final_predictions_rf, labels=np.unique(y2_test))
plot_confusion_matrix(cm_2, (10, 7), np.unique(y2_test), 'Level 2 Confusion Matrix')