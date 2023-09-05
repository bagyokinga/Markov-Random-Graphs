# -*- coding: utf-8 -*-
"""
Investigating the stability and chromatic numbers of Markov random graphs and
the associated lower and upper bounds
"""

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import milp
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

# Generate Markov Random Graphs

def generate_random_graphs(n,p,r):
    """
    Generates a Markov random graph.
    
    Parameters
    ----------
    n : int
        Number of vertices of the graph.
    p : float
        Initial edge probability.
        The probability that edge (1, j) exists for j > 1.
    r : float
        Reduction factor.
        P{edge (i+1, j) exists | edge (i, j) exists} = r * P{edge (i, j) exists}.

    Returns
    -------
    A : np array of size n x n
        Adjacency matrix: entry (i,j) gives the whether an edge exists between
        vertices i and j (value is 1 or 0). Note that the matrix is upper
        triangular.
        
    """
    # Set up n x n probability matrix: entry (i,j) gives the probability that
    # an edge exists between vertices i and j
    # note that it is upper triangular, not symmetric
    P = np.zeros((n,n))
    
    # Set up n x n adjacency matrix (upper triangular)
    A = np.zeros((n,n))
    
    # Set probability of edge (1,i) to p for all i > 1
    P[0] = np.ones(n) * p
    P[0,0] = 0
    
    # Generate Markov random matrix
    for i in range(n):
        for j in range(i, n):
            if np.random.rand() < P[i,j]:
                A[i,j] = 1
                # Apply reduction factor to next row of P if edge exists
                if i < n-1 and j > i+1: P[i+1, j] = r * P[i,j]
            else:
                # Keep next row of P the same if edge does not exist
                if i < n-1 and j > i+1: P[i+1, j] = P[i,j]
    
    return A


def edge_matrix(A):
    """
    Creates a matrix that has a row for each edge in the graph

    Parameters
    ----------
    A : np array of size n x n
        Adjacency matrix: entry (i,j) gives the whether an edge exists between
        vertices i and j (value is 1 or 0). Note that the matrix is upper
        triangular.

    Returns
    -------
    E : numpy array of size m x n
        Here n is the number of vertices, m is the number of edges in the graph.
        Edge matrix: each row corresponds to an edge. The column nr of the two
        entries in each row with value 1 denote the vertices connected by an edge.

    """
    n = len(A)      # number of vertices
    
    # Count number of edges
    m = 0 
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] == 1:
                m += 1
    
    # Create edge matrix from A
    E = np.zeros((m,n))
    count = 0
    for i in range(n):
        for j in range(i+1,n):
            # if there is an entry of 1 in A, add a row to E to represent an edge
            if A[i,j] == 1:
                E[count, i] = 1
                E[count, j] = 1
                count += 1        

    return E


# Finding the stability number of a graph

def alpha(E):
    """
    Find the stability number of a graph from its edge connectivity.
    
    Parameters
    ----------
    E : numpy array of size m x n
        Here n is the number of vertices, m is the number of edges in the graph.
        Edge matrix: each row corresponds to an edge. The column nr of the two
        entries in each row with value 1 denote the vertices connected by an edge.
        
    Returns
    -------
    a : int
        The stability number, a.k.a. independent set number. The size of the
        largest independent set in the graph.
    """
    
    # n = number of vertices in the graph
    n = len(E[0])
    # m = number of edges in the graph = nr of rows of E
    m = len(E)
    
    # milp optimization
    c = -np.ones(n)
    
    # for each edge (i,j), we need x_i + x_j <= 1
    b_u = np.ones(m)
    b_l = np.zeros(m)
    
    constraints = LinearConstraint(E, b_l, b_u)
    bounds = Bounds(lb=0, ub=1)     # 0 <= x_i <= 1
    integrality = np.ones_like(c)
    
    # define milp to be optimized
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    a = int(np.sum(res.x))
    
    return a


def alpha_gurobi(E):
    """
    Find the stability number of a graph from its edge connectivity.
    
    Parameters
    ----------
    E : numpy array of size m x n
        Here n is the number of vertices, m is the number of edges in the graph.
        Edge matrix: each row corresponds to an edge. The column nr of the two
        entries in each row with value 1 denote the vertices connected by an edge.
        
    Returns
    -------
    a : int
        The stability number, a.k.a. independent set number. The size of the
        largest independent set in the graph.
    """
    
    # n = number of vertices in the graph
    n = len(E[0])
    # m = number of edges in the graph = nr of rows of E
    m = len(E)
    
    # Create model
    model = gp.Model("MarkovGraphs")
    model.Params.LogToConsole = 0
    
    # Create variables
    x = model.addMVar(shape=n, vtype=GRB.BINARY, name="x")
    
    # Set objective
    obj = np.ones(n)
    model.setObjective(obj @ x, GRB.MAXIMIZE)
    
    # Build rhs vector
    rhs = np.ones(m)

    # Add constraints
    model.addConstr(E @ x <= rhs, name="c")
    
    # Optimize model
    model.optimize()
    
    a = np.sum(x.X)
    
    return a


# Lower bound by Greedy algorithm

def Greedy1(A):
    """
    Use Greedy algorithm to obtain a lower bound of the stability number, i.e.
    going from v_1 to v_n (descending vertex degrees w.h.p.) add the next vertex
    to the set if it is not connected to any of the vertices already included
    in the set.

    Parameters
    ----------
    A : numpy array of size n x n
        The adjacency matrix of a graph.

    Returns
    -------
    The number of elements of the independent set found.
    
    """
    
    n = len(A[0])       # number of vertices in the graph
    indep_set = [0]
    
    # add next vertex to set if it is not connected to any of the vertices in set
    for i in range(1,n):
        
        # create a variable that determines if next vertex is independent from
        # the existing elements of the set
        isIndep = True
        for v in indep_set:
            if A[v,i] == 1:
                isIndep = False
        
        # if vertex is independent, add it to the set
        if isIndep == True:
            indep_set.append(i)
            
    return len(indep_set)


def Greedy2(A):
    """
    Use Greedy algorithm to obtain a lower bound of the stability number, i.e.
    sorting vertices in ascending order by their vertex degrees, we add the
    next vertex to the set if it is not connected to any of the vertices already
    included in the set.

    Parameters
    ----------
    A : numpy array of size n x n
        The adjacency matrix of a graph.

    Returns
    -------
    The number of elements of the independent set found.
    
    """
    
    n = len(A[0])       # number of vertices in the graph
    
    # make A symmetric
    for i in range(1,n):
        for j in range(i):
            A[i,j] = A[j,i] 
    
    # add up the rows of A to get the vertex degrees
    vertex_degs = np.zeros(n)
    for i in range(n):
        vertex_degs[i] = np.sum(A[i])
    
    # sort the vertices by vertex degrees
    vertices = np.array(range(n))
    sorted_vertices = []
    
    for i in range(n):
        j = np.argmin(vertex_degs)
        sorted_vertices.append(vertices[j])
        vertex_degs[j] = n
    
    # add next vertex to set if it is not connected to any of the vertices in set
    indep_set = []
    for i in sorted_vertices:
        
        # create a variable that determines if next vertex is independent from
        # the existing elements of the set
        isIndep = True
        for v in indep_set:
            if A[v,i] == 1:
                isIndep = False
                
        # if vertex is independent, add it to the set
        if isIndep == True:
            indep_set.append(i)
            
    return len(indep_set)


# Spectral lower bounds

def spectral_lb_18(A):
    """
    Finds a spectral lower bound for independent set number based on Yildrim,
    inequality (18).
    
    Parameters
    ----------
    A : numpy array of size n x n
        The adjacency matrix of a graph.
        
    Returns
    -------
    A lower bound for the independent set number.

    """
    n = len(A[0])       # number of vertices
    
    # make A symmetric
    for i in range(1,n):
        for j in range(i):
            A[i,j] = A[j,i]
                        
    # create adjacency matrix of complement of A
    A_comp = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if A[i,j] == 0:
                    A_comp[i,j] = 1
                        
    # check connectivity
    G_comp = nx.from_numpy_matrix(A_comp)
    if nx.is_connected(G_comp) == True:
        
        # define eigenvalues and eigenvectors
        eigenvals, eigenvects = np.linalg.eig(A_comp)
        
        # sort eigenvalues in descending order (with corresp. eigenvects)
        ind = eigenvals.argsort()[::-1]      
        l = eigenvals[ind].real
        eigenvects = eigenvects.transpose()
        u = eigenvects[ind].real
        
        # create vector s = e^T*u_i
        s = np.zeros(n)
        for i in range(n):
            s[i] = np.sum(u[i])
        
        # create vector d
        d = np.zeros((n,n))
        for i in range(n):
            d[i] = u[i] - s[i]/s[0]*u[0]
        
        # Lower and upper bounds: tao_l <= tao <= tao_u
        tao_l = np.zeros(n)
        tao_u = np.zeros(n)
        for i in range(1,n):
            pos_lbounds = [-np.inf]
            pos_ubounds = [np.inf]
            for j in range(n):
                if -u[0,j]/(s[0]*d[i,j]) < 0 and d[i,j] > 0:
                    pos_lbounds.append(-u[0,j]/(s[0]*d[i,j]))
                if -u[0,j]/(s[0]*d[i,j]) > 0 and d[i,j] < 0:
                    pos_ubounds.append(-u[0,j]/(s[0]*d[i,j]))
            tao_l[i] = max(pos_lbounds)
            tao_u[i] = min(pos_ubounds) 
        
        # create vector v
        v = np.zeros(n)
        for i in range(1,n):
            
            # check that bounds were found
            if tao_l[i] != -np.inf and tao_u[i] != np.inf:
                
                maximum_found = False
                
                # avoid division by zero
                if l[i]*s[0]**2 + l[0]*s[i]**2 != 0:
                    
                    # set tao to be the value where 1st derivative is 0
                    tao = l[0]*s[i] / (l[i]*s[0]**2 + l[0]*s[i]**2)
                    
                    # check if tao is between the bounds for tao
                    if tao_l[i] < tao and tao < tao_u[i]:
                        
                        # check that it is a maximum point using 2nd derivative
                        if 2*(l[i]+l[0]*s[i]**2/s[0]**2) < 0:
                                        
                            v[i] = l[0]/s[0]**2 - (2*l[0]*s[i]/s[0]**2) * tao 
                            + (l[i]+l[0]*s[i]**2/s[0]**2) * tao**2
                            
                            maximum_found = True
                            
                
                # If the maximum of the parabola was not found, check endpoints
                if maximum_found == False:
                    
                    # left endpoint determined by tao_l
                    leftend_val = l[0]/s[0]**2 - (2*l[0]*s[i]/s[0]**2) * tao_l[i] 
                    + (l[i]+l[0]*s[i]**2/s[0]**2) * tao_l[i]**2
                    
                    # right endpoint determined by tao_u
                    rightend_val = l[0]/s[0]**2 - (2*l[0]*s[i]/s[0]**2) * tao_u[i]
                    + (l[i]+l[0]*s[i]**2/s[0]**2) * tao_u[i]**2
                    
                    # set v[i] to be the max of the two values
                    v[i] = max(leftend_val, rightend_val)

        v_star = max(v[1:])

        return 1/(1-v_star)
    
    else:
        return None

def spectral_lb_26(A):
    """
    Finds a spectral lower bound for independent set number based on Yildrim,
    inequality (26).
    
    Parameters
    ----------
    A : numpy array of size n x n
        The adjacency matrix of a graph.
        
    Returns
    -------
    A lower bound for the independent set number.

    """
    n = len(A[0])       # number of vertices
    
    # make A symmetric
    for i in range(1,n):
        for j in range(i):
            A[i,j] = A[j,i] 
            
    # create adjacency matrix of complement of A, count nr of edges
    A_comp = np.zeros((n,n))
    m = 0 
    for i in range(n):
        for j in range(n):
            if i != j:
                if A[i,j] == 1:
                    A_comp[i,j] = 0
                else:
                    A_comp[i,j] = 1
                    m += 1
    
    m = m/2     # edges were double counted

    # check connectivity
    G_comp = nx.from_numpy_matrix(A_comp)
    if nx.is_connected(G_comp) == True:
       
        # find lambda_n, the smallest eigenvalue
        min_eigval = np.min(np.linalg.eig(A_comp)[0].real)

        # return the lower bound given in the paper
        return 1+2*m/((n-2*m/n)*(2*m/n-min_eigval))
    
    # return nothing if graph is not connected
    else:
        return None


# Chromatic number of a graph

# Greedy for chromatic number

def chromatic_Greedy(A):
    """
    Use Greedy algorithm to obtain a lower bound of the chromatic number, i.e.
    going from v_1 to v_n (descending vertex degrees w.h.p.) add the next vertex
    to the set if it is not connected to any of the vertices already included
    in the set.

    Parameters
    ----------
    A : numpy array of size n x n
        The adjacency matrix of a graph.

    Returns
    -------
    The chromatic number of the graph.
    
    """
    
    n = len(A[0])       # number of vertices in the graph
    
    # Array denoting colours of vertices
    V = np.zeros(n)
    # Assign colour 1 to vertex 1
    V[0] = 1
    
    # Assign the 'smallest' available colour to other vertices
    for i in range(1,n):
        possible_colors = list(range(1,n+1))
        # Eliminate colours of previous adjacent vertices
        for j in range(i):
            if A[j,i] == 1:
                if V[j] in possible_colors:
                    possible_colors.remove(V[j])
        V[i] = min(possible_colors)
    
    return int(np.max(V))