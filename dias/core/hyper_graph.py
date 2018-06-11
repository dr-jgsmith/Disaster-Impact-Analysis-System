from numba import jit
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
"""
These functions use the Numba JIT compiler to speed up computational performance on
large data arrays and matrices. These are particularly useful for massive matrix multiplication
problems.

This is a work in progress!!! Changes will be made...

"""

# Optional: Step 0
# Note this step can be repeated
def computePattern(traffic_pattern_list, incidence):
    """
    In Q-Analysis it is often the case that we want to analyze the dynamics that occur on a representation/backcloth
    This is computed by multiplying the pattern vector on the incidence matrix of 0s and 1s.
    Here the pattern vector correspond to the vertices.
    
    convert vector weights into a numpy array
    
    :param traffic_pattern_list: ordered list of values
    :param incidence: numpy matrix
    :return: numpy matrix
    """
    try:
        vweights = np.array(traffic_pattern_list)
        # convert matrix to a numpy array
        vmatrix = np.array(incidence)
        # multiply the v-weights and numpy array (matrix)
        new_matrix = vweights * vmatrix
        # The new_matrix can now be processed again by passing the matrix to the incidentMatrix method
        return new_matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def addCoverPattern(cover_array, incidence):
    """
    This function is used to add a new dimension of relation to a given matrix composed of simplicies and vertices.
    This is accomplished by the addition between two vectors.
    
    :param cover_array: numpy array
    :param incidence: numpy array
    :return: numpy array
    """
    try:
        cover = np.array(cover_array)
        matrix = np.array(incidence)
        new_matrix = cover + matrix
        return new_matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass

@jit
def invert_pattern(pattern_vector):
    """
    Takes a pattern vector in binarized format and inverts the pattern
    :param pattern_vector: numpy array | 0's and 1's
    :return: numpy array
    """
    inverted = np.zeros(len(pattern_vector))
    for i in range(len(pattern_vector)):
        if pattern_vector[i] == 0.0:
            inverted[i] = 1.0
        else:
            inverted[i] = 0.0
    return inverted


def sparse_graph(incidence, hyperedge_list, theta):
    """
    This function encodes a sparse matrix into a graph representation. 
    This function provides a speed up in computation over the numpy matrix methods
    It can be used on both a shared face matrix or raw data input. 
    :param incidence: numpy incidence matrix 
    :param hyperedge_list: python list | simplicies or nodes
    :param theta: int
    :return: list of tuples
    """
    try:
        sparse = np.nonzero(incidence)
        edges = [(simplex, vertex, incidence[simplex][vertex]) for simplex, vertex in zip(sparse[0], sparse[1]) if incidence[simplex][vertex] > float(theta)]
        return edges
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def conjugate_graph(edges):
    """
    Takes a sparse graph consisting of edges between simplicies and vertices
    :param edges: sparse graph expressed as a list of tuples (simplex, vertex, value/dimension)
    :return: list of tuples
    """
    try:
        conjugate_edges = [(vertex, simplex, val) for vertex, simplex, val in edges]
        return conjugate_edges
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def dowker_relation(sparse_graph, conjugate_graph):
    """
    This provides a fast approach to computing the shared-face relation between simplicies.
    The function takes a sparse graph and its conjugate as inputs. Returns the dwoker relation + 1.
    To compute the true relation, subtract the -1 from the relation.
    
    :param sparse_graph: list of tuples
    :param conjugate_graph: list of tuples
    :return: numpy matrix
    """
    try:
        sparseg = compute_class_matrix_sparse(sparse_graph)
        conjq = compute_class_matrix_sparse(conjugate_graph)
        q_matrix = sparseg.dot(conjq).toarray()
        return q_matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def simple_ecc(array):
    """
    Compute the eccentricity of a simplex. Produced by computing the Dowker relation
    This takes a 1d array, which is typically the simplex and q-near vertices

    :param array: numpy array
    :return: numpy array
    """
    loc = array.argmax()
    new = np.delete(array, loc)
    qhat = max(array)-1
    qbottom = max(new)-1
    ecc = (qhat - qbottom) / (qbottom + 1)
    return ecc


def eccentricity(qmatrix):
    """
    takes a q-matrix computed from the dowker relation function
    :param qmatrix: numpy array
    :return: list of eccentricity values
    """
    try:
        # iterate through the matrix to compute the
        eccs = [simple_ecc(i) for i in qmatrix]
        return eccs
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def compute_classes(edges):
    """
    Collect all connected components - Identify equivelence classes
    These are the q-connected components.
    This is central data type for exploring multi-dimensional persistence of Eq Classes
    Takes a list of tuple edges
    :param edges: sparse graph 
    :return: list of sets
    """
    try:
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        comp = nx.connected_components(G)
        return  sorted(list(comp))
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def compute_class_graph(comp_list):
    """
    The ith value in the graph repreents the simplicial complex, will the jth value represents the simplex it is attached to the dimension.
    :param comp_list: a list of sets representing connected simplicies
    :return: graph representation with component indexed by location in the complex set
    """
    try:
        class_graph = [(i, j, 1) for i in range(len(comp_list)) for j in comp_list[i]]
        return class_graph
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def compute_class_matrix(sparse_graph):
    """
    Takes the constructed graph of a matrix of simplicial complexes and computes the sparse repreentation
    :param class_graph: list of tuples
    :return: dense matrix
    """
    try:
        row = np.array([i[0] for i in sparse_graph])
        col = np.array([i[1] for i in sparse_graph])
        data = np.array([i[2] for i in sparse_graph])
        matrix = csr_matrix((data, (row, col)), shape=(max(row)+1, max(col)+1 )).toarray()
        return matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def compute_class_matrix_sparse(sparse_graph):
    """
    Takes the constructed graph of a matrix of simplicial complexes and computes the sparse representation
    :param class_graph: list of tuples
    :return sparse matrix
    """
    try:
        row = np.array([i[0] for i in sparse_graph])
        col = np.array([i[1] for i in sparse_graph])
        data = np.array([i[2] for i in sparse_graph])
        matrix = csr_matrix((data, (row, col)), shape=(max(row) + 1, max(col) + 1))
        return matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass

@jit
def compute_paths(sparse_matrix, simplex_index):
    seen = np.array([simplex_index])
    fronts = [np.array([simplex_index])]
    for i in fronts:
        tmp = []
        for j in i:
            data = sparse_matrix.getrow(j).nonzero()[1]
            new = np.setdiff1d(data, seen)
            seen = np.union1d(new, seen)
            tmp.extend(new)

        if len(np.unique(tmp)) > 0:
            fronts.append(np.unique(tmp))
        else:
            pass
    return fronts


def sum_class_matrix(matrix, axis_val):
    sums = np.sum(matrix, axis=axis_val)
    return sums


def compute_qgraph(matrix, hyperedges, theta_list):
    qgraph = []
    try:
        for i in theta_list:
            edges = sparse_graph(matrix, hyperedges, i)
            cmp = compute_class_graph(edges)
            qgraph.append((i, cmp))
        print('length of graph: ', len(qgraph))
        return qgraph
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass

@jit
def compute_q_structure(qgraph):
    qstruct = [(j[0], len(j[1])) for j in qgraph]
    return qstruct

@jit
def compute_p_structure(qgraph):
    pstruct = ['x']
    for j in qgraph:
        tmp = []
        for i in j[1]:
            tmp.append(sum(i))
        pstruct.append((j[0], sum(tmp)))
    return pstruct[1:]



# Compute the eccentricity of each simplex in the complex.
# This measures how integrated a simplex is in the complex.
# Note: it would be interesting to compute a similar diagnostic for simplex persistence at each q-dim.
def computeEcc(EqClasses, qstruct, hyperedge_set, vertex_set, conjugate=False):
    if conjugate is False:
        hyperedges = hyperedge_set
    else:
        hyperedges = vertex_set

    # eccI = 2(sum(q_dim/num_simps))/(q_dim*(q_dim+1))
    eccentricity = {}
    for simplex in hyperedges:
        simplex_dim = []
        dim = 0
        for i in EqClasses:
            for j in i:
                if simplex in j:
                    val = dim / len(j)
                    simplex_dim.append(val)
                else:
                    pass
                dim = dim + 1
        # The ecc algorithm is based on the equation provided by Chin et al. 1991
        ecc = sum(simplex_dim) / ((1 / 2 * float(max(qstruct))) * float((max(qstruct) + 1)))
        eccentricity[simplex] = ecc
    return eccentricity


# Compute system complexity.
# This is only one of many possible complexity measures and is provided by John Casti.
def computeComplexity(q_percolation):
    strct = []
    vect = []
    for i in q_percolation.items():
        x = i[0] + 1
        y = x * i[1]
        strct.append(y)
        vect.append(i[1])
    z = sum(strct)
    complexity = 2 * (z / ((max(vect) + 1) * (max(vect) + 2)))
    return complexity
