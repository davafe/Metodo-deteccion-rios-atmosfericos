"""
A module for calculating component sizes in thresholded images with Elder Rule application.

This module provides functionality to analyze connected components in binary images,
applying thresholds and using the Elder Rule to track valid components across thresholds.

Functions:
    calculate_component_sizes_with_elder: Calculates relative sizes of valid components 
        across thresholds using Elder Rule

Dependencies:
    numpy as np
    cv2 (OpenCV)
"""

# Imports
import numpy as np
import cv2

# Function calculate_component_sizes_with_elder
def calculate_component_sizes_with_elder(minus_iwp, thresholds, hawai_matrix, america, verbose=0):
    """
    Calculate relative sizes of valid components of a matrix across thresholds using Elder Rule and cv2.THRESH_BINARY_INV
    
    A valid component is a connected region in the thresholded image that intersects
    both the Hawaii and America regions as defined by the input matrices.
    
    Args:
        minus_iwp: Input matrix to threshold (2D numpy array)
        thresholds: List of threshold values to apply (list of floats)
        hawai_matrix: Binary matrix representing Hawaii region (2D numpy array)
        america: Binary matrix representing America region (2D numpy array)
        verbose: Control output verbosity (0: silent, 1: basic, 2: detailed)
    
    Returns:
        List of relative component sizes (as fraction of total elements) for each threshold
    
    Note:
        Uses OpenCV's connectedComponents with 4-connectivity
        Applies Elder Rule to track components across thresholds
    """


    rows, cols = minus_iwp.shape
    num_elements = rows * cols
    component_sizes = []
    elder_component = None  # To track the first valid unique component

    # Iterate through all thresholds
    for threshold in thresholds:
        # Apply threshold to minus_iwp matrix
        _, thresh_image = cv2.threshold(minus_iwp, threshold, 255, cv2.THRESH_BINARY_INV)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(thresh_image.astype(np.uint8), 
                                                     connectivity=4
                                                     )
        if verbose>1:
            print(threshold, "number of conected components=", num_labels-1) # subtract 1 because component 0 is "background"     
        # Use np.unique to get component labels and their indices
        component_to_points = {i: set() for i in range(1, num_labels)} # Start from 1 because 0 is background
        for label in range(1, num_labels):   # Iterate only over labels (excluding background)
            # Get indices of points belonging to the label
            points = np.argwhere(labels == label)
            component_to_points[label] = set(map(tuple, points))  # Convert points to tuples (x, y)

        # Validate connected components
        valid_components = []   # Store valid components
        for component_points in component_to_points.values():
            # Check intersection with regions
            intersects_hawai = any(hawai_matrix[x, y] for x, y in component_points)
            intersects_america = any(america[x, y] for x, y in component_points)

            # If component intersects both regions, add to valid components
            if intersects_hawai and intersects_america:
                valid_components.append(component_points)

        # Apply Elder Rule
        if not valid_components:
            component_sizes.append(0)  # No valid components
            if verbose>0:
                print("With threshold=", threshold, "adds", 0, "to the vector")

        else:
            if elder_component is None:
                # Identify elder_component for first threshold with a single valid component
                if len(valid_components) == 1:
                    elder_component = valid_components[0]
                elif len(valid_components) > 1:
                    # Choose largest valid component if multiple exist
                    elder_component = max(valid_components, key=len)
                    if verbose>0:
                        print("Happened")
                if verbose>0:
                    print("With threshold=", threshold, "the function sets an elder component of size", len(elder_component))



            if elder_component is not None:
                # Find valid component that contains elder_component
                for component in valid_components:
                    if elder_component.issubset(component):
                        relative_size = len(component) / num_elements
                        component_sizes.append(relative_size)
                        if verbose>0:
                            print("With threshold=", threshold, "component_size=", len(component), "and adds:", relative_size,"to the vector")
                        break


    return component_sizes


