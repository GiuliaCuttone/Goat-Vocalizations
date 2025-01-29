"""
Goat vocalizations
Università degli studi di Milano

@author: Giulia Cuttone
"""

import pandas as pd 

# Z-score normalization (Standardization)
from sklearn.preprocessing import StandardScaler


'---------------------------------------------------------------------------------------------------------------'

def map_to_main_class(value, mapping):
    
    """
    Takes a subclass value and a mapping dictionary,
    and returns the corresponding main class if it exists in the mapping.
    If the subclass is not found, it returns the original value.
    ---------------------------------------------------------------------
    Args:
        value (str): The subclass to be mapped.
        mapping (dict): A dictionary mapping main classes to their subclasses.
    ----------------------------------------------------------------------------
    Returns:
        value (str): The main class corresponding to the subclass or the original value.
    """
    
    for main_class, subclasses in mapping.items():
        if value in subclasses:
            return main_class
    return value


def safe_assign_label(label):
    
    """
    Error handling
    --------------
    Paramenters:
        label (string): Audio file label.
    -------------------------------------
    Return:
        (string)
        'Unknown' if the label is not valid.
    """
    
    try:
        return assign_label(label)
    except ValueError:
        invalid_labels.append(label)
        return 'Unknown'


def assign_label(label):
    
    """
    Check the label and raise an exception if it's neither positive nor negative.
    -----------------------------------------------------------------------------
    Paramenters:
        label (string): Audio file label.
    -------------------------------------
    Returns:
        (string)
        'positive' if the label is in the positive labels list,
        'negative' if in the negative labels list.
    -----------------------------------------------------------
    Raises:
        ValueError: If the label is not found in either the positive or negative labels list.  
    """
    
    if label in Positive:
        return 'Positive'
    elif label in Negative:
        return 'Negative'
    else:
        raise ValueError(f"Label '{label}' is invalid: it must be in positive_labels or negative_labels")

'---------------------------------------------------------------------------------------------------------------'


"Main code:"

# Load dataset
file_path = './Vocapra_dataset.csv'
df = pd.read_csv(file_path)

# Exclude 'Class' column from normalization
features = df.drop(columns='Class')
scaler = StandardScaler()

# Apply Z-score normalization
normalized_features = scaler.fit_transform(features)


# Define the mapping of subclasses to main class
class_mapping = {
    'Calori': ['calori artificiali', 'calori naturali'],
    'Distribuzione Cibo': ['Distribuzione fieno', 'distribuzione concentrato', 'distribuzione unifeed'],
    'Fenomeni legati al parto': ['doglie del parto', 'fase espulsiva', 'parto difficile', 'aborto'],
    'Ferita-Morte': ['Ferita', 'Morte capra']
    }

Positive = ['Calori', 'Distribuzione Cibo', 'Presenza contemporanea di madri e capretti', 'Visita di estranei']
Negative = ['Fenomeni legati al parto', 'Ferita-Morte', 'Isolamento sociale', 'Separazione madre capretto']


# Initialize a list to keep track of invalid labels
invalid_labels = []

# Create a new DataFrame for the normalized data, preserving 'Class'
processed_data = pd.DataFrame(normalized_features, columns=features.columns)

# Replace subclasses with the main class
processed_data['Class'] = df['Class'].apply(lambda x: map_to_main_class(x, class_mapping))

# Add 'Emotional_state' column
processed_data['Emotional_state'] = processed_data['Class'].apply(safe_assign_label)

# Report invalid labels if there are any
if invalid_labels:
    print(f"\nInvalid labels encountered: {set(invalid_labels)}\n")


# Show the first few rows of the post-processed dataset
print(processed_data.head())

# Save the DataFrame to a CSV file
processed_data.to_csv('Vocapra_postprocessing.csv', index=False)
