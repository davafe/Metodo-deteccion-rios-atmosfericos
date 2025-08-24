"""
A module for generating boolean matrices based on geographical coordinates and pivot points.

This module provides functionality to create boolean masks based on latitude/longitude
conditions defined by pivot points and linear boundaries.

Functions:
    obtain_boolean_matrix: Creates a boolean matrix based on latitude/longitude conditions
        defined by three pivot points

Dependencies:
    numpy as np
"""

import numpy as np

def obtain_boolean_matrix(lats_used, lons_used, pivot1, pivot2, pivot3):
    """
    Create a boolean matrix based on geographical coordinates and pivot points.
    
    The function divides the latitude range into four regions using three pivot points
    and applies different longitude conditions for each region:
    1. Above pivot1: Linear boundary condition
    2. Between pivot1 and pivot2: Fixed longitude boundary
    3. Between pivot2 and pivot3: Different linear boundary condition
    4. Below pivot3: All False
    
    Args:
        lats_used: Array of latitude values (1D numpy array)
        lons_used: Array of longitude values (1D numpy array)
        pivot1: First latitude pivot point (float)
        pivot2: Second latitude pivot point (float)
        pivot3: Third latitude pivot point (float)
    
    Returns:
        Boolean matrix where True values indicate locations satisfying the conditions
        (2D numpy array of shape (len(lats_used), len(lons_used)))
    
    Note:
        The function uses vectorized operations for efficient computation
        The boundary conditions are based on linear equations in latitude-longitude space
    """
    # Create matrix initialized with False
    n = lats_used.shape[0]
    matrix = np.zeros((n, lons_used.shape[0]), dtype=bool)
    
    # Conditions for each latitude range
    condicion1 = (lats_used > pivot1)
    condicion2 = (lats_used > pivot2) & (lats_used <= pivot1)
    condicion3 = (lats_used <= pivot2) & (lats_used >= pivot3)
    condicion4 = (lats_used < pivot3)
    
    # Apply conditions to fill matrix rows
    matrix[condicion1, :] = lons_used > -1.411 * lats_used[condicion1][:, None] - 58.465
    matrix[condicion2, :] = lons_used > -125.5
    matrix[condicion3, :] = lons_used > -0.758 * lats_used[condicion3][:, None] - 95.1132
    matrix[condicion4, :] = False # equivalent to lons_used > lon_max
    
    return matrix