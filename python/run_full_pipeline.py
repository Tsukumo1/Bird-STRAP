import sys
import os

from bppr_data_processor import BPPRDataProcessor
from strap_embedding import STRAPEmbedding


def run_full_pipeline(
    bppr_result_dir,
    n_users,
    n_items,
    graph_name='avito',
    algo_name='BDPush',
    epsilon=0.5,
    embedding_dim=128,
    processed_data_dir='./processed_data',
    output_dir='./embeddings',
    ppr_threshold=0.0
):
    
    epsilon_str = str(epsilon)
    
    
    processor = BPPRDataProcessor(
        result_dir=bppr_result_dir,
        n_users=n_users,
        n_items=n_items,
        output_dir=processed_data_dir,
        graph_name=graph_name,
        algo_name=algo_name,
        epsilon_str=epsilon_str
    )


    if not os.listdir(processor.output_dir):
        P_merged, metadata = processor.run_full_pipeline(threshold=ppr_threshold)
        print(f"processed data save at: {processor.output_dir}/")
    else:
        print(f"processed data already existed: {processor.output_dir}/")
    
    
    strap = STRAPEmbedding(
        input_dir=processor.output_dir,
        epsilon=epsilon
    )
    
    emb_source, emb_target, sigma, emb_metadata = strap.run_strap_pipeline(
        d=embedding_dim,
        output_dir=output_dir,
        graph_name=graph_name,
        algo_name=algo_name,
        epsilon_str=epsilon_str
    )
    
    final_output_dir = os.path.join(output_dir, graph_name, algo_name, epsilon_str, str(embedding_dim))

    return {
        'user_embedding': emb_source[:n_users],
        'item_embedding': emb_source[n_users:],
        'metadata': emb_metadata,
        'singular_values': sigma,
        'output_dir': final_output_dir
    }

if __name__ == "__main__":
    config = {
        'bppr_result_dir': '../result/relative/ml-100k/BDPush/0.05',
        'n_users': 943,
        'n_items': 1682,
        'graph_name': 'ml-100k-0.5',
        'algo_name': 'BDPush',
        'epsilon': 0.0005,
        'embedding_dim': 128,          
        'processed_data_dir': '../processed_data',
        'output_dir': '../embeddings',
        'ppr_threshold': 0.0005/2
    }
    
    results = run_full_pipeline(**config)

    print(f"输出目录: {results['output_dir']}")