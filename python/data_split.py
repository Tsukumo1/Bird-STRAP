from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict

def prepare_test_data_from_graph(graph_file, n_users, n_items, test_ratio=0.2):
    edges = []
    edge_set = set()
    node_to_edges = defaultdict(list)
    
    with open(graph_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                u, v, w = int(parts[0])-1, int(parts[1])-1, float(parts[2])
            elif len(parts) == 2:
                u, v, w = int(parts[0])-1, int(parts[1])-1, 1.0
            edges.append((u, v, w))
            edge_set.add((u, v))
            node_to_edges[u].append((u, v, w))
            node_to_edges[v].append((u, v, w))
    
    print(f"总边数: {len(edges)}")
    
    train_edges = set()
    remaining_edges = set(edges)
    
    for node, node_edges in node_to_edges.items():
        found_in_train = False
        for e in node_edges:
            if e in remaining_edges:
                train_edges.add(e)
                remaining_edges.remove(e)
                found_in_train = True
                break
        if not found_in_train:
            train_edges.add(node_edges[0])
            remaining_edges.discard(node_edges[0])
    

    remaining_edges = list(remaining_edges)
    

    if len(remaining_edges) > 0:
        rem_train, test_edges_positive = train_test_split(
            remaining_edges, test_size=test_ratio, random_state=42
        )
        train_edges.update(rem_train)
    else:
        test_edges_positive = []
    
    train_edges = list(train_edges)
    
    test_edges_negative = []
    np.random.seed(42)
    attempts = 0
    while len(test_edges_negative) < len(test_edges_positive):
        u = np.random.randint(0, n_users)
        v = np.random.randint(0, n_items)
        attempts += 1
        if (u, v) not in edge_set:
            test_edges_negative.append((u, v, 0.0))
            edge_set.add((u, v))
        if attempts > len(test_edges_positive) * n_users * n_items * 10:
            break
    
    test_edges = test_edges_positive + test_edges_negative
    test_labels = np.array([1] * len(test_edges_positive) + [0] * len(test_edges_negative))

    
    train_file = graph_file.replace('u.data', 'graph.txt.new')
    test_file = graph_file.replace('u.data', 'graph_test.txt')

    with open(train_file, 'w') as f:
        for u, v, w in train_edges:
            f.write(f"{u} {v} {w}\n")
        
    with open(test_file, 'w') as f:
        for u, v, w in test_edges:
            f.write(f"{u} {v} {w}\n")

    return train_edges, test_edges, test_labels



if __name__ == "__main__":
    train_edges, test_edges, test_labels = prepare_test_data_from_graph(
        graph_file='../data/ml-100k/u.data', n_users=943, n_items=1682, test_ratio=0.5)