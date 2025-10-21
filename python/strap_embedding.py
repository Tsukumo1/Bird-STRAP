import numpy as np
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.linalg import svds
import json
import os
import argparse
from datetime import datetime


class STRAPEmbedding:
    def __init__(self, input_dir, epsilon=0.5):
        self.input_dir = input_dir
        self.epsilon = epsilon
        
        metadata_path = f"{input_dir}/metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        matrix_path = f"{input_dir}/proximity_matrix.npz"
        self.P = load_npz(matrix_path)

    
    def log_transform(self):
        if not isinstance(self.P, csr_matrix):
            P = self.P.tocsr()
        else:
            P = self.P.copy()
        
        # log(2/ε · P) = log(2/ε) + log(P)
        coefficient = 2.0 / self.epsilon
        P.data = np.log(coefficient * P.data)
        
        return P
    
    def compute_svd(self, P, d=128, which='LM'):
        U, Sigma, Vt = svds(P, k=d, which=which)
        idx = np.argsort(Sigma)[::-1]
        U = U[:, idx]
        Sigma = Sigma[idx]
        Vt = Vt[idx, :]
    
        return U, Sigma, Vt
            
    
    def generate_embeddings(self, U, Sigma, Vt):

        sqrt_sigma = np.sqrt(Sigma)
        embedding_source = U @ np.diag(sqrt_sigma)
        embedding_target = Vt.T @ np.diag(sqrt_sigma)

        return embedding_source, embedding_target
    
    def save_embeddings(self, embedding_source, embedding_target, Sigma, dim,output_dir='./embeddings',
                       graph_name=None, algo_name=None, epsilon_str=None):

        if graph_name and algo_name and epsilon_str:
            output_dir = os.path.join(output_dir, graph_name, algo_name, epsilon_str, str(dim))
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        source_path = f"{output_dir}/embedding_source.npy"
        target_path = f"{output_dir}/embedding_target.npy"
        sigma_path = f"{output_dir}/singular_values.npy"
        
        np.save(source_path, embedding_source)
        np.save(target_path, embedding_target)
        np.save(sigma_path, Sigma)
        
        emb_metadata = {
            'timestamp': timestamp,
            'epsilon': self.epsilon,
            'embedding_dim': embedding_source.shape[1],
            'n_nodes': embedding_source.shape[0],
            'n_users': self.metadata['n_users'],
            'n_items': self.metadata['n_items'],
            'top_10_singular_values': Sigma[:10].tolist(),
            'source_embedding_norm': float(np.linalg.norm(embedding_source)),
            'target_embedding_norm': float(np.linalg.norm(embedding_target)),
        }
        
        metadata_path = f"{output_dir}/embedding_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(emb_metadata, f, indent=2)
        

        n_users = self.metadata['n_users']
        user_embedding = embedding_source[:n_users]
        item_embedding = embedding_source[n_users:]
        
        user_path = f"{output_dir}/u_embedding.npy"
        item_path = f"{output_dir}/v_embedding.npy"
        
        np.save(user_path, user_embedding)
        np.save(item_path, item_embedding)
        
        return emb_metadata
    
    def run_strap_pipeline(self, d=128, output_dir='./embeddings',
                          graph_name=None, algo_name=None, epsilon_str=None):
        P_log = self.log_transform()
        U, Sigma, Vt = self.compute_svd(P_log, d=d)
        embedding_source, embedding_target = self.generate_embeddings(U, Sigma, Vt)
        metadata = self.save_embeddings(embedding_source, embedding_target, Sigma, d,
                                       output_dir, graph_name, algo_name, epsilon_str)
        
        return embedding_source, embedding_target, Sigma, metadata


def main():
    parser = argparse.ArgumentParser(description='STRAP Embedding Generator')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory of intermediate data')
    parser.add_argument('--output_dir', type=str, default='./embeddings',
                        help='Root directory for output')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='Epsilon parameter for STRAP')
    parser.add_argument('--dim', type=int, default=128,
                        help='Dimension of the embedding')
    parser.add_argument('--graph_name', type=str, default=None,
                        help='Name of the graph, used for building output path')
    parser.add_argument('--algo_name', type=str, default=None,
                        help='Name of the algorithm, used for building output path')
    parser.add_argument('--epsilon_str', type=str, default=None,
                        help='Epsilon, used for building output path)') 
    
    args = parser.parse_args()
    
    strap = STRAPEmbedding(
        input_dir=args.input_dir,
        epsilon=args.epsilon
    )
    
    emb_source, emb_target, sigma, metadata = strap.run_strap_pipeline(
        d=args.dim,
        output_dir=args.output_dir,
        graph_name=args.graph_name,
        algo_name=args.algo_name,
        epsilon_str=args.epsilon_str
    )
    
    


if __name__ == "__main__":
    main()