import numpy as np
import networkx as nx

def label_to_state (label, digits):
    return np.array(map(int,list(format(label,'0'+str(digits)+'b'))))

def state_to_label (state):
    return int(''.join(map(str,state)),2)

# (label of the input state) -> [transition] -> (label of the output state)

def pseudo_transition(matrix,thresholds,label):
    return np.matmul(matrix.transpose(),label_to_state(label,len(matrix)))-thresholds

# "pseudo_transition" gives the result of the linear part of the operation: A^T . X - B

def sgn1(array0,pseudo_array1):
    array1 = []
    for i in range(0,len(array0)):
        if pseudo_array1[i] > 0:
            state = 1
        else: state = 0
        array1.append(state)        
    return np.array(array1)

def sgn0(array0,pseudo_array1):
    array1 = []
    for i in range(0,len(array0)):
        if pseudo_array1[i] == 0:
            state = array0[i]
        elif pseudo_array1[i] > 0:
            state = 1
        else:
            state = 0            
        array1.append(state)        
    return np.array(array1)

# The introduction of the 'sign_convention' parameter is the 'transition' function is the main difference 
# between 'nee1' and 'neet2' It selects the 'sgn' function.
# 0 corresponds to the 'null' convention, like in Hyunju's paper.
# 1 corresponds to the' positive' convention, like in Fumia & Martins. 

def transition(matrix,thresholds,sign_convention,label):
    if sign_convention == 0:
        newlabel = state_to_label(
            sgn0(label_to_state(label,len(matrix)),pseudo_transition(matrix,thresholds,label))
        )
    else:
        newlabel = state_to_label(
            sgn1(label_to_state(label,len(matrix)),pseudo_transition(matrix,thresholds,label))
        )
    return newlabel

def find_attractors(transition_matrix):

    S = nx.Graph()
    DS = nx.DiGraph()
    
    T = transition_matrix
    nodes = range(len(T))
    
    edges = []
    for i in range(0,len(T)):
        edges.append((T[i,0],T[i,1]))
    
    S.add_nodes_from(nodes)
    S.add_edges_from(edges)
    DS.add_nodes_from(nodes)
    DS.add_edges_from(edges)
    
    att_list = list(nx.simple_cycles(DS))
    
    basin = []
    for i in range(len(att_list)):
        basin.append(len(nx.node_connected_component(S,att_list[i][0])))
    
    attractors = []
    for i in range(len(att_list)):
        attractors.append([att_list[i],basin[i]])
    
    return attractors