"""
JAX-based operations for DIAS core functionality.

This module provides JAX JIT-compiled functions for high-performance
numerical computations in disaster impact analysis.
"""

from typing import Tuple

import jax
import jax.numpy as jnp


# ============================================================================
# Basic Array Operations
# ============================================================================


@jax.jit
def invert_pattern(pattern_vector: jnp.ndarray) -> jnp.ndarray:
    """
    Invert a binary pattern vector.
    
    Converts 0s to 1s and 1s to 0s.
    
    Args:
        pattern_vector: Binary array of shape (n,)
    
    Returns:
        Inverted pattern array of shape (n,)
    
    Example:
        >>> pattern = jnp.array([1.0, 0.0, 1.0, 0.0])
        >>> inverted = invert_pattern(pattern)
        >>> print(inverted)
        [0. 1. 0. 1.]
    """
    return 1.0 - pattern_vector


@jax.jit
def invert_matrix(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Invert all rows in a matrix using vectorized operations.
    
    Args:
        matrix: Binary matrix of shape (m, n)
    
    Returns:
        Inverted matrix of shape (m, n)
    
    Example:
        >>> matrix = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        >>> inverted = invert_matrix(matrix)
        >>> print(inverted)
        [[0. 1.]
         [1. 0.]]
    """
    return 1.0 - matrix


@jax.jit
def normalize_matrix(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize matrix columns to [0, 1] range using min-max normalization.
    
    Each column is independently normalized:
    normalized = (x - min(column)) / (max(column) - min(column))
    
    Args:
        matrix: Input matrix of shape (m, n)
    
    Returns:
        Normalized matrix of shape (m, n)
    
    Example:
        >>> matrix = jnp.array([[1.0, 10.0], [5.0, 20.0]])
        >>> normalized = normalize_matrix(matrix)
        >>> print(normalized)
        [[0.   0.  ]
         [1.   1.  ]]
    """
    col_min = jnp.min(matrix, axis=0, keepdims=True)
    col_max = jnp.max(matrix, axis=0, keepdims=True)
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-10
    denominator = col_max - col_min + epsilon
    
    return (matrix - col_min) / denominator


@jax.jit
def compute_incident_matrix(
    value_matrix: jnp.ndarray, theta: float, slice_type: str = "upper"
) -> jnp.ndarray:
    """
    Compute binary incidence matrix based on threshold.
    
    Args:
        value_matrix: Input matrix of shape (m, n)
        theta: Threshold value
        slice_type: "upper" for >= threshold, "lower" for <= threshold
    
    Returns:
        Binary incidence matrix of shape (m, n)
    
    Example:
        >>> values = jnp.array([[1.0, 5.0], [3.0, 7.0]])
        >>> incident = compute_incident_matrix(values, 4.0, "upper")
        >>> print(incident)
        [[0 1]
         [0 1]]
    """
    if slice_type == "upper":
        result = value_matrix >= theta
    else:
        result = value_matrix <= theta
    
    return result.astype(jnp.int32)


# ============================================================================
# Distance Calculations
# ============================================================================


@jax.jit
def euclidean_distance_pairwise(
    coords1: jnp.ndarray, coords2: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of coordinates.
    
    Args:
        coords1: Coordinates of shape (n, d) where d is dimension
        coords2: Coordinates of shape (m, d)
    
    Returns:
        Distance matrix of shape (n, m)
    
    Example:
        >>> coords1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> coords2 = jnp.array([[1.0, 0.0]])
        >>> distances = euclidean_distance_pairwise(coords1, coords2)
        >>> print(distances)
        [[1.        ]
         [1.41421356]]
    """
    # Expand dimensions for broadcasting: (n, 1, d) - (1, m, d) -> (n, m, d)
    diff = coords1[:, None, :] - coords2[None, :, :]
    squared_distances = jnp.sum(diff**2, axis=-1)
    return jnp.sqrt(squared_distances)


@jax.jit
def haversine_distance(
    lat1: jnp.ndarray,
    lon1: jnp.ndarray,
    lat2: jnp.ndarray,
    lon2: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate geodesic distance using Haversine formula.
    
    Computes the great-circle distance between points on Earth's surface.
    
    Args:
        lat1: Latitude of first point(s) in decimal degrees
        lon1: Longitude of first point(s) in decimal degrees
        lat2: Latitude of second point(s) in decimal degrees
        lon2: Longitude of second point(s) in decimal degrees
    
    Returns:
        Distance in meters
    
    Example:
        >>> # Houston to Galveston (approximately)
        >>> lat1 = jnp.array([29.7604])
        >>> lon1 = jnp.array([-95.3698])
        >>> lat2 = jnp.array([29.3013])
        >>> lon2 = jnp.array([-94.7977])
        >>> dist = haversine_distance(lat1, lon1, lat2, lon2)
        >>> print(f"{dist[0]/1000:.1f} km")
        72.4 km
    """
    # Earth's radius in meters
    R = 6371000.0
    
    # Convert to radians
    lat1_rad = jnp.radians(lat1)
    lat2_rad = jnp.radians(lat2)
    dlat = jnp.radians(lat2 - lat1)
    dlon = jnp.radians(lon2 - lon1)
    
    # Haversine formula
    a = (
        jnp.sin(dlat / 2.0) ** 2
        + jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * jnp.sin(dlon / 2.0) ** 2
    )
    c = 2 * jnp.arcsin(jnp.sqrt(a))
    
    return R * c


@jax.jit
def haversine_distance_matrix(
    coords1: jnp.ndarray, coords2: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute pairwise geodesic distances between coordinate sets.
    
    Args:
        coords1: Coordinates of shape (n, 2) where columns are [lat, lon]
        coords2: Coordinates of shape (m, 2) where columns are [lat, lon]
    
    Returns:
        Distance matrix of shape (n, m) in meters
    
    Example:
        >>> coords1 = jnp.array([[29.7604, -95.3698]])  # Houston
        >>> coords2 = jnp.array([[29.7505, -95.3605]])  # Nearby point
        >>> distances = haversine_distance_matrix(coords1, coords2)
        >>> print(f"{distances[0, 0]:.0f} meters")
        1523 meters
    """
    # Extract lat/lon and expand for broadcasting
    lat1 = coords1[:, 0:1]  # Shape: (n, 1)
    lon1 = coords1[:, 1:2]  # Shape: (n, 1)
    lat2 = coords2[:, 0:1].T  # Shape: (1, m)
    lon2 = coords2[:, 1:2].T  # Shape: (1, m)
    
    # Compute pairwise distances
    return haversine_distance(lat1, lon1, lat2, lon2)


# ============================================================================
# Connectivity Operations
# ============================================================================


@jax.jit
def build_adjacency_matrix(
    distances: jnp.ndarray, threshold: float
) -> jnp.ndarray:
    """
    Build binary adjacency matrix from distance matrix.
    
    Two nodes are adjacent if their distance is below the threshold.
    
    Args:
        distances: Distance matrix of shape (n, n)
        threshold: Maximum distance for adjacency
    
    Returns:
        Binary adjacency matrix of shape (n, n)
    
    Example:
        >>> distances = jnp.array([[0.0, 1.0, 5.0],
        ...                        [1.0, 0.0, 2.0],
        ...                        [5.0, 2.0, 0.0]])
        >>> adj = build_adjacency_matrix(distances, 3.0)
        >>> print(adj)
        [[1 1 0]
         [1 1 1]
         [0 1 1]]
    """
    # Nodes within threshold are adjacent
    adjacent = distances <= threshold
    return adjacent.astype(jnp.int32)


@jax.jit
def compute_connectivity(
    elevations: jnp.ndarray,
    adjacency: jnp.ndarray,
    water_level: float,
) -> jnp.ndarray:
    """
    Compute connectivity matrix for flood inundation.
    
    Two adjacent parcels are connected if both are below water level.
    
    Args:
        elevations: Elevation array of shape (n,)
        adjacency: Adjacency matrix of shape (n, n)
        water_level: Water level threshold
    
    Returns:
        Connectivity matrix of shape (n, n)
    
    Example:
        >>> elevations = jnp.array([5.0, 10.0, 8.0])
        >>> adjacency = jnp.ones((3, 3), dtype=jnp.int32)
        >>> connectivity = compute_connectivity(elevations, adjacency, 9.0)
        >>> # Parcels 0 and 2 are both flooded (< 9.0) and adjacent
    """
    # Parcels below water level are flooded
    flooded = (elevations < water_level).astype(jnp.int32)
    
    # Pairwise flooding: both parcels must be flooded
    flooded_pairs = flooded[:, None] * flooded[None, :]
    
    # Connected if adjacent AND both flooded
    connected = adjacency * flooded_pairs
    
    return connected


# ============================================================================
# Statistical Operations
# ============================================================================


@jax.jit
def weighted_average(values: jnp.ndarray, weights: jnp.ndarray) -> float:
    """
    Compute weighted average of values.
    
    Args:
        values: Value array of shape (n,)
        weights: Weight array of shape (n,)
    
    Returns:
        Weighted average
    
    Example:
        >>> values = jnp.array([1.0, 2.0, 3.0])
        >>> weights = jnp.array([0.5, 0.3, 0.2])
        >>> avg = weighted_average(values, weights)
        >>> print(f"{avg:.2f}")
        1.70
    """
    return jnp.sum(values * weights) / jnp.sum(weights)


@jax.jit
def compute_percentile_mask(
    values: jnp.ndarray, percentile: float
) -> jnp.ndarray:
    """
    Create binary mask for values above a percentile threshold.
    
    Args:
        values: Value array of shape (n,)
        percentile: Percentile threshold (0-100)
    
    Returns:
        Binary mask of shape (n,)
    
    Example:
        >>> values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> mask = compute_percentile_mask(values, 60)
        >>> print(mask)
        [0 0 1 1 1]
    """
    threshold = jnp.percentile(values, percentile)
    return (values >= threshold).astype(jnp.int32)


# ============================================================================
# Impact and Value Propagation
# ============================================================================


@jax.jit
def propagate_impact(
    values: jnp.ndarray,
    connectivity: jnp.ndarray,
    impact_factor: float,
) -> jnp.ndarray:
    """
    Propagate impact through connected parcels.
    
    Each parcel's value is reduced by impact from connected parcels.
    
    Args:
        values: Current parcel values of shape (n,)
        connectivity: Connectivity matrix of shape (n, n)
        impact_factor: Multiplier for impact (0-1)
    
    Returns:
        Updated values of shape (n,)
    
    Example:
        >>> values = jnp.array([100.0, 100.0, 100.0])
        >>> connectivity = jnp.array([[1, 1, 0],
        ...                           [1, 1, 1],
        ...                           [0, 1, 1]])
        >>> new_values = propagate_impact(values, connectivity, 0.1)
        >>> # Each parcel loses value based on connected parcels
    """
    # Number of connections for each parcel
    n_connections = jnp.sum(connectivity, axis=1)
    
    # Average impact from connections
    impact = (n_connections * impact_factor).reshape(-1)
    
    # Reduce values by impact
    new_values = values * (1.0 - impact)
    
    # Ensure non-negative
    return jnp.maximum(new_values, 0.0)


@jax.jit
def compute_cumulative_impact(
    initial_values: jnp.ndarray,
    connectivity_matrices: jnp.ndarray,
    impact_factor: float,
) -> jnp.ndarray:
    """
    Compute cumulative impact over multiple time steps.
    
    Args:
        initial_values: Starting values of shape (n,)
        connectivity_matrices: Connectivity for each step, shape (t, n, n)
        impact_factor: Impact multiplier per step
    
    Returns:
        Final values after all impacts of shape (n,)
    
    Example:
        >>> initial = jnp.array([100.0, 100.0])
        >>> connectivity = jnp.array([[[1, 1], [1, 1]]])  # One time step
        >>> final = compute_cumulative_impact(initial, connectivity, 0.1)
    """
    def step_fn(values: jnp.ndarray, connectivity: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
        new_values = propagate_impact(values, connectivity, impact_factor)
        return new_values, None
    
    final_values, _ = jax.lax.scan(step_fn, initial_values, connectivity_matrices)
    return final_values


# ============================================================================
# Random Number Generation (with explicit PRNG keys)
# ============================================================================


def generate_random_events(
    key: jax.random.PRNGKey,
    n_events: int,
    intensity_range: Tuple[float, float],
) -> jnp.ndarray:
    """
    Generate random event intensities.
    
    Note: Not JIT-compiled as it takes a PRNG key.
    
    Args:
        key: JAX PRNG key
        n_events: Number of events to generate
        intensity_range: (min_intensity, max_intensity)
    
    Returns:
        Array of random intensities of shape (n_events,)
    
    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> events = generate_random_events(key, 5, (1.0, 10.0))
        >>> print(events.shape)
        (5,)
    """
    min_intensity, max_intensity = intensity_range
    return jax.random.uniform(
        key, shape=(n_events,), minval=min_intensity, maxval=max_intensity
    )


# ============================================================================
# Array Updates (JAX immutable array handling)
# ============================================================================


@jax.jit
def update_array_at_indices(
    array: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray
) -> jnp.ndarray:
    """
    Update array at specific indices (JAX immutable approach).
    
    Args:
        array: Input array of shape (n,)
        indices: Indices to update
        values: New values
    
    Returns:
        Updated array (new copy)
    
    Example:
        >>> arr = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2])
        >>> values = jnp.array([10.0, 30.0])
        >>> updated = update_array_at_indices(arr, indices, values)
        >>> print(updated)
        [10.  2. 30.  4.]
    """
    return array.at[indices].set(values)


@jax.jit
def multiply_array_at_indices(
    array: jnp.ndarray, indices: jnp.ndarray, multiplier: float
) -> jnp.ndarray:
    """
    Multiply array values at specific indices.
    
    Args:
        array: Input array of shape (n,)
        indices: Indices to multiply
        multiplier: Multiplication factor
    
    Returns:
        Updated array (new copy)
    
    Example:
        >>> arr = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([1, 3])
        >>> updated = multiply_array_at_indices(arr, indices, 2.0)
        >>> print(updated)
        [1. 4. 3. 8.]
    """
    return array.at[indices].multiply(multiplier)


# ============================================================================
# Vectorized Operations for Batch Processing
# ============================================================================


# Vectorize haversine distance for batch processing
haversine_distance_vectorized = jax.vmap(
    haversine_distance, in_axes=(0, 0, 0, 0)
)

# Vectorize pattern inversion for batch processing
invert_pattern_vectorized = jax.vmap(invert_pattern)

# Vectorize impact propagation for batch processing
propagate_impact_vectorized = jax.vmap(
    propagate_impact, in_axes=(0, None, None)
)

