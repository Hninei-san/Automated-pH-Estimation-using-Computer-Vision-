import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from itertools import permutations

def generate_color_sequences(colors):
    """
    Generate all possible order sequences of the given colors.
    
    Args:
    - colors (numpy.ndarray): Array containing RGB values for each color.
    
    Returns:
    - sequences (list of tuples): List containing all possible order sequences of the colors.
    """
    # Generate all permutations of the colors
    color_permutations = permutations(colors)
    
    # Convert permutations to list of tuples
    sequences = list(color_permutations)
    
    return sequences

def predict_ph_value_with_distance(new_test_strip, n_neighbors=3):
    """
    Predict the pH value of a test strip with unsorted colors using K-nearest neighbors.
    
    Args:
    - new_test_strip (numpy.ndarray): Test strip with unsorted colors.
    - n_neighbors (int): Number of neighbors to use in K-nearest neighbors (default is 3).
    
    Returns:
    - predicted_pH_value (float): Predicted pH value for the test strip.
    - nearest_neighbor_distance (float): Distance to the nearest neighbor for the predicted pH value.
    - color_sequence (list): The sequence of colors used for prediction.
    """
    # Load data from the .txt file
    data = np.loadtxt('pHdata.txt', delimiter=',')
    X_train = data[:, :-1]  # Features (RGB values), all columns except the last one
    y_train = data[:, -1]   # Labels (pH values), last column
    
    # Initialize the classifier (K-nearest neighbors)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the classifier on the predefined test strips
    knn.fit(X_train, y_train)

    # Generate all possible order sequences
    sequences = generate_color_sequences(new_test_strip)

    # Initialize list to store predicted pH values, their distances, and color sequences
    predictions = []

    # Iterate through each sequence
    for sequence in sequences:
        # Extract features from the sequence
        X_test = np.array(sequence).reshape(1, -1)

        # Predict the pH value of the sequence
        predicted_pH = knn.predict(X_test)

        # Obtain the distance to the nearest neighbor
        nearest_neighbor_distance = knn.kneighbors(X_test)[0][0][0]

        # Append predicted pH value, its distance, and the sequence to the list
        predictions.append((predicted_pH[0], nearest_neighbor_distance, sequence))

    # Sort predictions based on distance
    predictions.sort(key=lambda x: x[1])

    # Return the predicted pH value, its nearest neighbor distance, and the sequence of colors
    return predictions[0]