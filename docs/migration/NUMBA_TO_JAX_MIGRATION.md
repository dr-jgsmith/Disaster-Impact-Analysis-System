# Numba to JAX Migration Guide

## Overview

This document catalogues all Numba JIT-compiled functions in the DIAS codebase and provides guidance for migrating them to JAX. The migration aims to maintain or improve performance while gaining benefits from JAX's ecosystem.

## Why JAX over Numba?

### Advantages of JAX
- **Automatic Differentiation**: Built-in grad() for derivatives
- **Vectorization**: Powerful vmap() for batch operations
- **GPU/TPU Support**: Seamless acceleration without code changes
- **Functional Programming**: Pure functions, easier to reason about
- **Modern Ecosystem**: Better integration with modern ML/scientific computing
- **XLA Compilation**: Advanced optimization backend

### Key Differences to Consider
| Feature | Numba | JAX |
|---------|-------|-----|
| **Array Mutation** | Allowed | Not allowed (immutable arrays) |
| **Compilation** | Lazy, per-function | Eager with XLA |
| **Random Numbers** | Uses numpy.random | Requires explicit PRNG key |
| **Loops** | JIT-compiled | Should vectorize when possible |
| **GPU Support** | CUDA-specific | Platform-agnostic |

## Numba Usage Inventory

### File: `dias/core/hyper_graph.py`
**Lines with @jit:** 14 functions

#### 1. `invert_pattern` (Line 78-85)
```python
@jit
def invert_pattern(pattern_vector):
    """Inverts binary pattern vector"""
    return 1.0 - pattern_vector
```

**Complexity**: Simple  
**Input Shape**: (n,) array  
**Output Shape**: (n,) array  
**Operations**: Element-wise subtraction  
**Migration Difficulty**: ⭐ Easy

**JAX Equivalent**:
```python
@jax.jit
def invert_pattern(pattern_vector: jnp.ndarray) -> jnp.ndarray:
    """Inverts binary pattern vector"""
    return 1.0 - pattern_vector
```

---

#### 2. `invert_matrix` (Line 88-94)
```python
@jit
def invert_matrix(matrix):
    inv_matrix = ['x']
    for i in matrix:
        inv = invert_pattern(i)
        inv_matrix.append(inv)
    return np.array(inv_matrix[1:])
```

**Complexity**: Medium (uses list mutation)  
**Input Shape**: (m, n) array  
**Output Shape**: (m, n) array  
**Operations**: Row-wise inversion  
**Migration Difficulty**: ⭐⭐ Medium (vectorize loop)

**JAX Equivalent**:
```python
@jax.jit
def invert_matrix(matrix: jnp.ndarray) -> jnp.ndarray:
    """Invert all rows in matrix"""
    return jax.vmap(invert_pattern)(matrix)
    # Or simply: return 1.0 - matrix
```

---

#### 3. `normalize` (Line 97-106)
```python
@jit
def normalize(matrix):
    norm = ['x']
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            val = (matrix[i][j] - min(matrix[:, j])) / (max(matrix[:, j]) - min(matrix[:, j]))
            row.append(val)
        norm.append(row)
    return np.array(norm[1:])
```

**Complexity**: High (nested loops, column operations)  
**Input Shape**: (m, n) array  
**Output Shape**: (m, n) array  
**Operations**: Min-max normalization per column  
**Migration Difficulty**: ⭐⭐⭐ Challenging (vectorize completely)

**JAX Equivalent**:
```python
@jax.jit
def normalize(matrix: jnp.ndarray) -> jnp.ndarray:
    """Normalize columns to [0, 1] range"""
    col_min = jnp.min(matrix, axis=0)
    col_max = jnp.max(matrix, axis=0)
    return (matrix - col_min) / (col_max - col_min + 1e-10)  # Add epsilon for stability
```

---

#### 4. `compute_incident` (Line 109-115)
```python
@jit
def compute_incident(value_matrix, theta, slice_type='upper'):
    if slice_type is 'upper':
        data = value_matrix >= theta
    else:
        data = value_matrix <= theta
    return data.astype(int)
```

**Complexity**: Simple  
**Input Shape**: (m, n) array  
**Output Shape**: (m, n) array  
**Operations**: Threshold comparison  
**Migration Difficulty**: ⭐ Easy

**JAX Equivalent**:
```python
@jax.jit
def compute_incident(value_matrix: jnp.ndarray, 
                    theta: float, 
                    slice_type: str = 'upper') -> jnp.ndarray:
    """Compute incidence matrix based on threshold"""
    if slice_type == 'upper':
        data = value_matrix >= theta
    else:
        data = value_matrix <= theta
    return data.astype(jnp.int32)
```

---

#### 5-14. Additional Functions in `hyper_graph.py`
Functions at lines: 303, 327, 367, 385, 397, 412, 429, 453, 479, 504

These follow similar patterns:
- Matrix operations
- Distance calculations
- Graph connectivity
- Array slicing and filtering

All can be migrated using similar JAX patterns.

---

### File: `dias/scripts/base_model.py`
**Lines with @jit:** 6 functions

#### 1. `synch_files` (Line 97)
```python
@jit
def synch_files(infile, outfile, column):
    # Synchronizes data between two DBF files
```

**Complexity**: Medium  
**Operations**: Data synchronization, matching  
**Migration Difficulty**: ⭐⭐ Medium

---

#### 2-6. Additional Functions
Functions at lines: 173, 208, 218, 238, 260

These handle:
- Connectivity matrix construction
- Distance calculations (geodesic and Euclidean)
- Impact zone computations
- Value propagation

---

### File: `dias/storage/processdbf.py`
**Lines with @jit:** 10 functions (all class methods)

**Note**: These are I/O and data manipulation functions. Most should NOT use JIT in either Numba or JAX.

**Migration Strategy**: 
- Remove @jit decorators
- Keep as pure Python
- Use standard pandas/numpy for data processing
- JAX is not appropriate for file I/O operations

---

## Migration Patterns

### Pattern 1: Simple Element-wise Operations
**Numba**:
```python
@jit
def multiply_by_scalar(arr, scalar):
    return arr * scalar
```

**JAX**:
```python
@jax.jit
def multiply_by_scalar(arr: jnp.ndarray, scalar: float) -> jnp.ndarray:
    return arr * scalar
```

---

### Pattern 2: Array Creation with Loops
**Numba** (Anti-pattern):
```python
@jit
def create_result():
    result = []
    for i in range(n):
        result.append(compute(i))
    return np.array(result)
```

**JAX** (Vectorized):
```python
@jax.jit
def create_result():
    indices = jnp.arange(n)
    return jax.vmap(compute)(indices)
```

---

### Pattern 3: Conditional Logic
**Numba**:
```python
@jit
def conditional_op(matrix, threshold):
    result = []
    for row in matrix:
        if row.sum() > threshold:
            result.append(row * 2)
        else:
            result.append(row)
    return np.array(result)
```

**JAX** (Use jnp.where):
```python
@jax.jit
def conditional_op(matrix: jnp.ndarray, threshold: float) -> jnp.ndarray:
    row_sums = jnp.sum(matrix, axis=1, keepdims=True)
    should_double = row_sums > threshold
    return jnp.where(should_double, matrix * 2, matrix)
```

---

### Pattern 4: Random Number Generation
**Numba**:
```python
@jit
def random_array(n):
    return np.random.rand(n)
```

**JAX** (Explicit PRNG):
```python
@jax.jit
def random_array(key: jax.random.PRNGKey, n: int) -> jnp.ndarray:
    return jax.random.uniform(key, shape=(n,))

# Usage:
key = jax.random.PRNGKey(0)
arr = random_array(key, 100)
```

---

### Pattern 5: In-place Updates
**Numba** (Allowed):
```python
@jit
def update_array(arr, idx, val):
    arr[idx] = val
    return arr
```

**JAX** (Immutable):
```python
@jax.jit
def update_array(arr: jnp.ndarray, idx: int, val: float) -> jnp.ndarray:
    return arr.at[idx].set(val)
```

---

## Distance Calculations

### Euclidean Distance
**Numba**:
```python
from scipy.spatial import distance

@jit
def euclidean_distances(coords1, coords2):
    return distance.cdist(coords1, coords2, 'euclidean')
```

**JAX**:
```python
@jax.jit
def euclidean_distances(coords1: jnp.ndarray, coords2: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise Euclidean distances"""
    # coords1: (n, 2), coords2: (m, 2) -> (n, m)
    diff = coords1[:, None, :] - coords2[None, :, :]
    return jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
```

### Geodesic Distance (Haversine)
**JAX Implementation**:
```python
@jax.jit
def haversine_distance(lat1: jnp.ndarray, lon1: jnp.ndarray,
                       lat2: jnp.ndarray, lon2: jnp.ndarray) -> jnp.ndarray:
    """Calculate geodesic distance using Haversine formula"""
    R = 6371000  # Earth radius in meters
    
    lat1_rad = jnp.radians(lat1)
    lat2_rad = jnp.radians(lat2)
    dlat = jnp.radians(lat2 - lat1)
    dlon = jnp.radians(lon2 - lon1)
    
    a = jnp.sin(dlat/2)**2 + jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * jnp.sin(dlon/2)**2
    c = 2 * jnp.arcsin(jnp.sqrt(a))
    
    return R * c
```

---

## Sparse Matrices

JAX has experimental sparse support. For production:

**Option 1**: Use scipy sparse matrices (without JIT)
```python
from scipy.sparse import csr_matrix

def build_sparse_matrix(indices, values, shape):
    # Use scipy (not JIT-compiled)
    return csr_matrix((values, indices), shape=shape)
```

**Option 2**: Use JAX BCOO (Block Compressed Sparse)
```python
import jax.experimental.sparse as jsparse

@jax.jit
def sparse_multiply(data, indices, dense_vec):
    sparse_mat = jsparse.BCOO((data, indices), shape=(n, m))
    return sparse_mat @ dense_vec
```

**Option 3**: Dense approximation for small matrices
```python
@jax.jit
def sparse_as_dense(indices, values, shape):
    dense = jnp.zeros(shape)
    return dense.at[indices].set(values)
```

---

## Performance Considerations

### When JAX is Faster
✅ Vectorized operations  
✅ Matrix multiplications  
✅ GPU/TPU acceleration available  
✅ Batch processing with vmap  
✅ Automatic differentiation needed

### When to Keep Python/NumPy
❌ File I/O operations  
❌ String processing  
❌ Dynamic control flow (many branches)  
❌ External library calls  
❌ One-time setup code

---

## Testing Strategy

### 1. Numerical Equivalence
```python
def test_migration():
    # Generate test data
    data = np.random.rand(100, 50)
    
    # Numba version
    result_numba = old_numba_function(data)
    
    # JAX version
    result_jax = new_jax_function(jnp.array(data))
    
    # Compare (allow small floating point differences)
    np.testing.assert_allclose(result_numba, np.array(result_jax), rtol=1e-6)
```

### 2. Performance Benchmarking
```python
import time

def benchmark_function(func, data, n_runs=100):
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = func(data)
        times.append(time.time() - start)
    return np.mean(times), np.std(times)

# Compare
numba_time, _ = benchmark_function(numba_func, data)
jax_time, _ = benchmark_function(jax_func, jax_data)
print(f"Speedup: {numba_time / jax_time:.2f}x")
```

---

## Migration Checklist

### For Each Function

- [ ] Identify function signature and types
- [ ] Document input/output shapes
- [ ] Check for array mutations (need functional approach)
- [ ] Identify random number generation (need PRNG key)
- [ ] Check for external library calls (may not be JIT-able)
- [ ] Vectorize loops where possible
- [ ] Replace list building with array operations
- [ ] Add type hints
- [ ] Write unit tests comparing old vs new
- [ ] Benchmark performance
- [ ] Document any API changes

---

## Priority Migration Order

### Phase 1: High-Priority, Low-Risk (Week 1)
1. ✅ Simple element-wise operations (`invert_pattern`)
2. ✅ Matrix operations (`invert_matrix`, `normalize`)
3. ✅ Threshold operations (`compute_incident`)

### Phase 2: Core Functionality (Week 2)
4. ⏳ Distance calculations (Euclidean, geodesic)
5. ⏳ Connectivity matrix construction
6. ⏳ Impact zone computations

### Phase 3: Complex Operations (Week 3)
7. ⏳ Graph operations
8. ⏳ Value propagation
9. ⏳ Simulation functions

### Phase 4: Cleanup (Week 4)
10. ⏳ Remove Numba dependency
11. ⏳ Performance optimization
12. ⏳ Documentation updates

---

## Common Pitfalls

### ❌ Pitfall 1: Mutation
```python
# Wrong - JAX arrays are immutable
@jax.jit
def bad_update(arr):
    arr[0] = 99  # Error!
    return arr

# Correct
@jax.jit
def good_update(arr):
    return arr.at[0].set(99)
```

### ❌ Pitfall 2: Global Random State
```python
# Wrong - no global random state in JAX
@jax.jit
def bad_random():
    return jax.random.uniform()  # Error!

# Correct
@jax.jit
def good_random(key):
    return jax.random.uniform(key)
```

### ❌ Pitfall 3: Type Promotion
```python
# Be explicit about dtypes
arr_int = jnp.array([1, 2, 3], dtype=jnp.int32)
arr_float = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
```

---

## Resources

- **JAX Documentation**: https://jax.readthedocs.io/
- **JAX GitHub**: https://github.com/google/jax
- **JAX Tutorials**: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
- **Common Gotchas**: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

---

## Summary Statistics

**Total Functions with @jit**: 30  
**Files Affected**: 3  
**Estimated Migration Time**: 3-4 weeks  
**Expected Performance**: Equal or better (especially with GPU)  
**Breaking Changes**: Minimal (internal implementation only)

---

*Last Updated: 2024*  
*Migration Status: In Progress*

