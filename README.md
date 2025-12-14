# STRAP Pipeline 
基于STRAP算法的图embedding生成与下游任务评估流程。

## 目录结构

```
├── data/                          # 数据目录
│   └── ml-100k/                   # MovieLens-100K数据集
│       ├── u.data                 # 原始图数据
│       ├── graph.txt.new          # 训练集
│       └── graph_test.txt         # 测试集
├── result/relative/               # BIRD生成的PPR文件目录
├── embeddings/                    # 生成的embedding输出目录
├── data_split.py                  # 数据切分脚本
├── bppr_data_processor.py         # PPR矩阵处理
├── strap_embedding.py             # STRAP embedding生成
├── run_full_pipeline.py           # 完整流程运行脚本
└── downstream_tasks.py            # 下游任务评估
```

## 数据准备

### 数据格式要求
与BIRD算法要求的格式一致，每行格式为：
```
user_id item_id [weight]
```

示例：
```
0 10 5.0
0 15 4.0
1 10 3.0
```

### 示例数据集
本项目使用 [MovieLens-100K](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset/data) 数据集：
- 用户给电影评分的二分图
- 包含权重信息（评分1-5）
- 适合Link Prediction和评分预测任务

## 使用流程

### Step 1: 数据切分

运行 `data_split.py` 将数据切分为训练集和测试集（50/50）：

```bash
python data_split.py
```

**修改参数：**
```python
# 在 data_split.py 中修改
graph_file = './data/ml-100k/u.data'  # 输入文件路径
```

**输出：**
- `graph.txt.new`: 训练集（包含所有节点，50%的边）
- `graph_test.txt`: 测试集（剩余50%的边，以及采样同等数量负样本）

**任务说明：**
- 针对 Link Prediction 任务设计
- 训练集包含图中所有节点，但只有部分边

---

### Step 2: 运行BIRD算法

运行BIRD算法生成PPR（Personalized PageRank）文件：

```bash
# 运行你的BIRD代码
# 输出PPR文件到 result/relative/ 目录
```
---

### Step 3: 生成Embedding

运行 `run_full_pipeline.py` 完成PPR的矩阵处理和embedding生成：

```bash
python run_full_pipeline.py
```

**修改参数：**
```python
# 在 run_full_pipeline.py 中修改
config = {
        'bppr_result_dir': '../result/relative/ml-100k/BDPush/0.05',
        'n_users': 943,
        'n_items': 1682,
        'graph_name': 'ml-100k-0.5',
        'algo_name': 'BDPush',
        'epsilon': 0.0005,
        'embedding_dim': 128,          
        'processed_data_dir': '../processed_data',   # bppr_data_processor.py的中期输出路径。（用于检查）
        'output_dir': '../embeddings',               # embedding的存储路径。
        'ppr_threshold': 0.0005/2
    }

```
---

### Step 4: 下游任务评估

运行 `downstream_tasks.py` 进行Link Prediction评估：

```bash
python downstream_tasks.py
```

**修改参数：**
```python
# 在 downstream_tasks.py 中修改
embedding_dir = './embeddings/ml-100k-0.5/BDPush/5e-05/128'  # embedding目录
test_file = './data/ml-100k/graph_test.txt'                  # 测试集路径
```

**当前支持的任务：**
-  Link Prediction（链接预测）
  - 评估指标：AUC, AP, Precision, Recall, F1-Score

**输出示例：**
```
最佳阈值: 3.7721 (F1=0.8340)

 分类指标 (阈值=3.7721):
  Precision: 0.7877
  Recall:    0.8861
  F1-Score:  0.8340

 排序指标:
  AUC: 0.9027
  AP:  0.8940

 数据统计:
  测试样本数: 98318
  正样本数:   49159 (50.00%)
  负样本数:   49159 (50.00%)
```
