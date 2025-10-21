"""
完整修复后的 BPPR Data Processor
修复：文件名从0开始，循环也应从0开始
"""

import numpy as np
from scipy.sparse import lil_matrix, save_npz
import os
import json
from tqdm import tqdm


class BPPRDataProcessor:
    def __init__(self, result_dir, n_users, n_items, output_dir='./processed_data', 
                 graph_name='avito', algo_name='BDPush', epsilon_str='0.5'):

        self.result_dir = result_dir
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items  # 总节点数
        
        self.output_dir = os.path.join(output_dir, graph_name, algo_name, epsilon_str)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    
    def get_global_node_id(self, node_id, is_item=False):
        if is_item:
            return self.n_users + node_id
        return node_id
    
    def read_ppr_file(self, filepath, threshold=0.0):
        ppr_dict = {}
        if not os.path.exists(filepath):
            return ppr_dict
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    node_id = int(parts[0])
                    ppr_value = float(parts[1])
                    
                    if ppr_value >= threshold:
                        ppr_dict[node_id] = ppr_value
        return ppr_dict
    
    def process_forward_ppr(self, threshold=0.0):
        
        P = lil_matrix((self.n_nodes, self.n_nodes))
        total_entries = 0
        missing_files = []
        
        for u in tqdm(range(self.n_users), desc="u->all nodes"):
            source_global = self.get_global_node_id(u, is_item=False)
            
            u2u_file = f"{self.result_dir}/{u}.txt"
            if not os.path.exists(u2u_file):
                missing_files.append(u2u_file)
            u2u_ppr = self.read_ppr_file(u2u_file, threshold)
            for target_local, ppr in u2u_ppr.items():
                target_global = self.get_global_node_id(target_local, is_item=False)
                P[source_global, target_global] = ppr
                total_entries += 1
            
            u2v_file = f"{self.result_dir}/{u}_v.txt"
            if not os.path.exists(u2v_file):
                missing_files.append(u2v_file)
            u2v_ppr = self.read_ppr_file(u2v_file, threshold)
            for target_local, ppr in u2v_ppr.items():
                target_global = self.get_global_node_id(target_local, is_item=True)
                P[source_global, target_global] = ppr
                total_entries += 1
        
        for v in tqdm(range(self.n_items), desc="v->all nodes"):
            source_global = self.get_global_node_id(v, is_item=True)

            v2v_file = f"{self.result_dir}/v_{v}.txt"
            if not os.path.exists(v2v_file):
                missing_files.append(v2v_file)
            v2v_ppr = self.read_ppr_file(v2v_file, threshold)
            for target_local, ppr in v2v_ppr.items():
                target_global = self.get_global_node_id(target_local, is_item=True)
                P[source_global, target_global] = ppr
                total_entries += 1

            v2u_file = f"{self.result_dir}/v_{v}_u.txt"
            if not os.path.exists(v2u_file):
                missing_files.append(v2u_file)
            v2u_ppr = self.read_ppr_file(v2u_file, threshold)
            for target_local, ppr in v2u_ppr.items():
                target_global = self.get_global_node_id(target_local, is_item=False)
                P[source_global, target_global] = ppr
                total_entries += 1
        
        return P
    
    def process_transpose_ppr(self, threshold=0.0):

        
        P_T = lil_matrix((self.n_nodes, self.n_nodes))
        total_entries = 0

        for u in tqdm(range(self.n_users), desc="Transpose u->all nodes"):
            source_global = self.get_global_node_id(u, is_item=False)
            
            u2u_file = f"{self.result_dir}/{u}.txt"
            u2u_ppr = self.read_ppr_file(u2u_file, threshold)
            for target_local, ppr in u2u_ppr.items():
                target_global = self.get_global_node_id(target_local, is_item=False)
                P_T[target_global, source_global] = ppr
                total_entries += 1
            
            u2v_file = f"{self.result_dir}/{u}_v.txt"
            u2v_ppr = self.read_ppr_file(u2v_file, threshold)
            for target_local, ppr in u2v_ppr.items():
                target_global = self.get_global_node_id(target_local, is_item=True)

                P_T[target_global, source_global] = ppr
                total_entries += 1
        

        for v in tqdm(range(self.n_items), desc="Transpose v->all nodes"):
            source_global = self.get_global_node_id(v, is_item=True)
            
            v2v_file = f"{self.result_dir}/v_{v}.txt"
            v2v_ppr = self.read_ppr_file(v2v_file, threshold)
            for target_local, ppr in v2v_ppr.items():
                target_global = self.get_global_node_id(target_local, is_item=True)
                P_T[target_global, source_global] = ppr
                total_entries += 1
            
            v2u_file = f"{self.result_dir}/v_{v}_u.txt"
            v2u_ppr = self.read_ppr_file(v2u_file, threshold)
            for target_local, ppr in v2u_ppr.items():
                target_global = self.get_global_node_id(target_local, is_item=False)
                P_T[target_global, source_global] = ppr
                total_entries += 1
        
        return P_T
    
    def merge_and_save(self, P, P_T):
        # P = P + P_T
        P_merged = P + P_T

        print(f"  矩阵非零元素: {P_merged.nnz}")
        print(f"  稀疏度: {P_merged.nnz / (self.n_nodes ** 2):.8f}")
        print(f"  矩阵形状: {P_merged.shape}")
        
        P_merged = P_merged.tocsr()
        matrix_path = f"{self.output_dir}/proximity_matrix.npz"
        save_npz(matrix_path, P_merged)
        
        metadata = {
            'n_nodes': self.n_nodes,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'nnz': P_merged.nnz,
            'sparsity': P_merged.nnz / (self.n_nodes ** 2),
            'shape': list(P_merged.shape),
            'format': 'csr',
            'description': 'Proximity matrix P from BPPR, merged from forward and transpose',
            'node_id_range': f'Users: 0-{self.n_users-1}, Items: {self.n_users}-{self.n_nodes-1}'
        }
        
        metadata_path = f"{self.output_dir}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return P_merged, metadata
    
    def run_full_pipeline(self, threshold=0.0):
        P = self.process_forward_ppr(threshold)
        P_T = self.process_transpose_ppr(threshold)
        P_merged, metadata = self.merge_and_save(P, P_T)
        return P_merged, metadata


if __name__ == "__main__":
    result_dir = "./result/relative/women/BDPush/0.5"
    n_users = 18
    n_items = 14
    
    output_dir = "./processed_data"
    graph_name = "women"
    algo_name = "BDPush"
    epsilon_str = "0.5"
    
    threshold = 0.0025
    

    processor = BPPRDataProcessor(
        result_dir=result_dir,
        n_users=n_users,
        n_items=n_items,
        output_dir=output_dir,
        graph_name=graph_name,
        algo_name=algo_name,
        epsilon_str=epsilon_str
    )
    

    P_merged, metadata = processor.run_full_pipeline(threshold=threshold)