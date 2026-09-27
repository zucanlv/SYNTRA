# 导入faiss库
import faiss
import numpy as np

# 检查是否使用GPU
print("是否支持GPU：", faiss.get_num_gpus() > 0)

# 简单测试GPU索引创建
if faiss.get_num_gpus() > 0:
    # 创建随机向量（1000个维度为128的向量）
    dim = 128
    num_vectors = 1000
    vectors = np.random.rand(num_vectors, dim).astype('float32')
    
    # 使用GPU创建索引（FlatL2：L2距离的暴力检索索引）
    res = faiss.StandardGpuResources()  # 初始化GPU资源
    cpu_index = faiss.IndexFlatL2(dim)  # 先创建CPU索引
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 转换为GPU索引
    
    # 添加向量到GPU索引
    gpu_index.add(vectors)
    
    # 检索测试（找前5个最相似的向量）
    query = np.random.rand(1, dim).astype('float32')
    distances, indices = gpu_index.search(query, 5)
    
    print("检索结果 - 距离：", distances)
    print("检索结果 - 索引：", indices)
else:
    print("未检测到GPU，faiss可能使用CPU模式（若仅安装faiss-gpu但无GPU，也可降级使用CPU）")