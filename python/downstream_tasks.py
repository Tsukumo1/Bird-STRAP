import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import json


class DownstreamTasks:
    def __init__(self, embedding_dir='./embeddings'):
        self.embedding_dir = embedding_dir
        
        self.user_emb = np.load(f'{embedding_dir}/u_embedding.npy')
        self.item_emb = np.load(f'{embedding_dir}/v_embedding.npy')
        
        with open(f'{embedding_dir}/embedding_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.n_users = self.user_emb.shape[0]
        self.n_items = self.item_emb.shape[0]
        self.embedding_dim = self.user_emb.shape[1]
        
    
    
    def predict_link_score(self, user_id, item_id, method='dot'):
        if method == 'dot':
            return np.dot(self.user_emb[user_id], self.item_emb[item_id])
        
        elif method == 'cosine':
            u = self.user_emb[user_id]
            v = self.item_emb[item_id]
            return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-10)
        
        elif method == 'hadamard':
            return np.linalg.norm(self.user_emb[user_id] * self.item_emb[item_id])
    
    def batch_predict(self, edges, method='dot'):
        scores = []
        for user_id, item_id in edges:
            score = self.predict_link_score(user_id, item_id, method)
            scores.append(score)
        return np.array(scores)
    
    def compute_precision_recall_f1(self, test_labels, pred_labels):
        precision = precision_score(test_labels, pred_labels, zero_division=0)
        recall = recall_score(test_labels, pred_labels, zero_division=0)
        f1 = f1_score(test_labels, pred_labels, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def find_best_threshold(self, test_labels, scores):
        precisions, recalls, thresholds = precision_recall_curve(test_labels, scores)
        
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        return best_threshold, best_f1
    
    def evaluate_link_prediction(self, test_edges, test_labels, method='dot', 
                                use_best_threshold=False):

        scores = self.batch_predict(test_edges, method)

        auc = roc_auc_score(test_labels, scores)
        ap = average_precision_score(test_labels, scores)
        
        if use_best_threshold:
            threshold, best_f1_from_curve = self.find_best_threshold(test_labels, scores)
            print(f"\n最佳阈值: {threshold:.4f} (F1={best_f1_from_curve:.4f})")
        else:
            threshold = 0.5
            print(f"\n使用默认阈值: {threshold}")
        
        pred_labels = (scores >= threshold).astype(int)
        prf_metrics = self.compute_precision_recall_f1(test_labels, pred_labels)

        print(f"\n 分类指标 (阈值={threshold:.4f}):")
        print(f"  Precision: {prf_metrics['precision']:.4f}")
        print(f"  Recall:    {prf_metrics['recall']:.4f}")
        print(f"  F1-Score:  {prf_metrics['f1']:.4f}")
        
        print(f"\n 排序指标:")
        print(f"  AUC: {auc:.4f}")
        print(f"  AP:  {ap:.4f}")
        
        print(f"\n 数据统计:")
        print(f"  测试样本数: {len(test_edges)}")
        print(f"  正样本数:   {np.sum(test_labels)} ({np.mean(test_labels)*100:.2f}%)")
        print(f"  负样本数:   {len(test_labels) - np.sum(test_labels)} ({(1-np.mean(test_labels))*100:.2f}%)")
        
        metrics = {
            'auc': auc,
            'ap': ap,
            'threshold': threshold,
            'method': method,
            'n_samples': len(test_edges),
            'n_positive': int(np.sum(test_labels)),
            **prf_metrics,
        }
        
        return metrics
    
    def load_test_data(self, file_path, task='link_predict'):
        test_edges = []
        test_labels = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                u = int(parts[0])
                v = int(parts[1])
                w = float(parts[2]) if len(parts) > 2 else 1.0
                
                test_edges.append((u, v))
                
                if task == 'link_predict':
                    label = 1 if w > 0 else 0
                else:
                    label = w
                test_labels.append(label)
        
        test_labels = np.array(test_labels, dtype=np.int32)
        return test_edges, test_labels


if __name__ == "__main__":
    tasks = DownstreamTasks(embedding_dir='../embeddings/ml-100k-0.5/BDPush/0.0005/128')
    
    test_edges, test_labels = tasks.load_test_data('../data/ml-100k/graph_test.txt')

    metrics = tasks.evaluate_link_prediction(
        test_edges, 
        test_labels, 
        method='dot',
        use_best_threshold=True
    )
    