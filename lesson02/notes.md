# 第2课：线性代数复习 I
# Lesson 2: Linear Algebra Review I

## 学习目标 (Learning Objectives)

通过本课学习，您将能够：
- 理解向量和矩阵的基本概念 (Understand basic concepts of vectors and matrices)
- 掌握向量运算的几何意义 (Master geometric meaning of vector operations)
- 熟练使用NumPy进行矩阵运算 (Proficiently use NumPy for matrix operations)
- 为机器学习中的数学运算打下基础 (Build foundation for mathematical operations in ML)

## 课程内容 (Course Content)

### 1. 向量基础 (Vector Fundamentals)

#### 向量的定义 (Vector Definition)
向量是具有大小和方向的量，在机器学习中用于表示数据点、特征等。
Vectors are quantities with magnitude and direction, used in ML to represent data points, features, etc.

```python
import numpy as np

# 向量表示 (Vector representation)
# 2D向量 (2D vector)
v1 = np.array([3, 4])      # 位置向量 (position vector)
v2 = np.array([-1, 2])     # 另一个向量 (another vector)

# 3D向量 (3D vector)
v3 = np.array([1, -2, 3])  # 三维空间中的向量 (vector in 3D space)

# 网络安全中的应用：特征向量 (Application in security: feature vector)
network_features = np.array([
    1500,    # 数据包大小 (packet size)
    0.5,     # 持续时间 (duration)
    80,      # 端口号 (port number)
    3        # 协议类型编码 (protocol type encoding)
])
```

#### 向量运算 (Vector Operations)

```python
# 向量加法 (Vector addition)
# 几何意义：向量的头尾相接 (Geometric meaning: head-to-tail connection)
v_sum = v1 + v2
print(f"v1 + v2 = {v_sum}")  # [2, 6]

# 向量减法 (Vector subtraction)
v_diff = v1 - v2
print(f"v1 - v2 = {v_diff}")  # [4, 2]

# 标量乘法 (Scalar multiplication)
# 改变向量的大小，不改变方向 (Changes magnitude, not direction)
v_scaled = 2 * v1
print(f"2 * v1 = {v_scaled}")  # [6, 8]
```

#### 向量的重要性质 (Important Vector Properties)

```python
# 向量长度（模长） (Vector magnitude/norm)
magnitude_v1 = np.linalg.norm(v1)
print(f"|v1| = {magnitude_v1}")  # 5.0

# 单位向量 (Unit vector)
unit_v1 = v1 / magnitude_v1
print(f"v1的单位向量 (unit vector): {unit_v1}")

# 点积 (Dot product)
dot_product = np.dot(v1, v2)
print(f"v1 · v2 = {dot_product}")  # 5

# 点积的几何意义：cosθ = (v1·v2)/(|v1||v2|)
# Geometric meaning: cosθ = (v1·v2)/(|v1||v2|)
```

### 2. 矩阵基础 (Matrix Fundamentals)

#### 矩阵的定义和表示 (Matrix Definition and Representation)

```python
# 矩阵创建 (Matrix creation)
# 2x3矩阵 (2x3 matrix)
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# 3x2矩阵 (3x2 matrix)
B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

print(f"矩阵A的形状 (Shape of A): {A.shape}")  # (2, 3)
print(f"矩阵B的形状 (Shape of B): {B.shape}")  # (3, 2)
```

#### 特殊矩阵 (Special Matrices)

```python
# 零矩阵 (Zero matrix)
zero_matrix = np.zeros((3, 3))

# 单位矩阵 (Identity matrix)
identity_matrix = np.eye(3)

# 对角矩阵 (Diagonal matrix)
diagonal_matrix = np.diag([1, 2, 3])

print("单位矩阵 (Identity matrix):")
print(identity_matrix)

print("对角矩阵 (Diagonal matrix):")
print(diagonal_matrix)
```

### 3. 矩阵运算 (Matrix Operations)

#### 基本矩阵运算 (Basic Matrix Operations)

```python
# 矩阵转置 (Matrix transpose)
A_transpose = A.T
print(f"A的转置 (A transpose):\n{A_transpose}")

# 矩阵加法（同型矩阵） (Matrix addition - same shape)
C = np.array([
    [1, 1, 1],
    [2, 2, 2]
])
matrix_sum = A + C
print(f"A + C =\n{matrix_sum}")

# 标量乘法 (Scalar multiplication)
scalar_mult = 2 * A
print(f"2 * A =\n{scalar_mult}")
```

#### 矩阵乘法 (Matrix Multiplication)

```python
# 矩阵乘法规则：(m,n) × (n,p) → (m,p)
# Matrix multiplication rule: (m,n) × (n,p) → (m,p)

# A是2x3，B是3x2，所以A×B是2x2
matrix_product = np.dot(A, B)
# 或者使用 @ 操作符 (or use @ operator)
matrix_product_alt = A @ B

print(f"A × B =\n{matrix_product}")

# 验证矩阵乘法的计算过程 (Verify matrix multiplication process)
print("矩阵乘法详细计算 (Detailed calculation):")
print(f"第一行第一列: {A[0, :]} · {B[:, 0]} = {np.dot(A[0, :], B[:, 0])}")
print(f"第一行第二列: {A[0, :]} · {B[:, 1]} = {np.dot(A[0, :], B[:, 1])}")
```

### 4. 线性代数在网络安全中的应用 (Linear Algebra Applications in Cybersecurity)

#### 特征向量表示 (Feature Vector Representation)

```python
# 网络流量特征矩阵 (Network traffic feature matrix)
# 每行代表一个连接，每列代表一个特征
# Each row represents a connection, each column represents a feature
traffic_features = np.array([
    [1500, 0.5, 80, 6],    # 连接1：大小,时长,端口,协议
    [800, 0.2, 443, 6],    # 连接2
    [64, 0.001, 53, 17],   # 连接3
    [2000, 1.5, 22, 6],    # 连接4
])

feature_names = ["数据包大小", "持续时间", "端口", "协议"]
print("网络流量特征矩阵 (Network traffic feature matrix):")
print(traffic_features)
```

#### 数据标准化 (Data Normalization)

```python
# Z-score标准化 (Z-score normalization)
def standardize_features(X):
    """标准化特征矩阵 (Standardize feature matrix)"""
    mean = np.mean(X, axis=0)  # 按列计算均值
    std = np.std(X, axis=0)    # 按列计算标准差
    return (X - mean) / std, mean, std

# 应用标准化 (Apply standardization)
standardized_features, feature_means, feature_stds = standardize_features(traffic_features)

print("标准化后的特征矩阵 (Standardized feature matrix):")
print(np.round(standardized_features, 2))

print(f"特征均值 (Feature means): {np.round(feature_means, 2)}")
print(f"特征标准差 (Feature std): {np.round(feature_stds, 2)}")
```

#### 距离计算 (Distance Calculation)

```python
# 欧几里得距离 (Euclidean distance)
def euclidean_distance(x1, x2):
    """计算两个向量间的欧几里得距离 (Calculate Euclidean distance)"""
    return np.sqrt(np.sum((x1 - x2)**2))

# 计算连接间的相似性 (Calculate similarity between connections)
connection1 = standardized_features[0]
connection2 = standardized_features[1]

distance = euclidean_distance(connection1, connection2)
print(f"连接1和连接2的距离 (Distance): {distance:.3f}")

# 批量计算所有连接对之间的距离 (Batch calculate all pairwise distances)
n_connections = len(standardized_features)
distance_matrix = np.zeros((n_connections, n_connections))

for i in range(n_connections):
    for j in range(n_connections):
        distance_matrix[i, j] = euclidean_distance(
            standardized_features[i], 
            standardized_features[j]
        )

print("距离矩阵 (Distance matrix):")
print(np.round(distance_matrix, 3))
```

### 5. 矩阵分解初步 (Introduction to Matrix Decomposition)

```python
# 简单的矩阵分解示例 (Simple matrix decomposition example)
# LU分解（用于线性方程组求解）(LU decomposition for solving linear systems)

from scipy.linalg import lu

# 创建一个方阵 (Create a square matrix)
square_matrix = np.array([
    [2, 1, 1],
    [1, 3, 2],
    [1, 0, 0]
], dtype=float)

# LU分解 (LU decomposition)
P, L, U = lu(square_matrix)

print("原矩阵 A (Original matrix):")
print(square_matrix)

print("下三角矩阵 L (Lower triangular):")
print(np.round(L, 3))

print("上三角矩阵 U (Upper triangular):")
print(np.round(U, 3))

# 验证分解结果 (Verify decomposition)
reconstructed = P @ L @ U
print("重构矩阵 P×L×U (Reconstructed):")
print(np.round(reconstructed, 3))
```

## 代码示例讲解 (Code Example Explanation)

本课的 `example.py` 文件包含了以下内容：

1. **向量运算演示** (Vector Operations Demo)：展示各种向量运算及其几何意义
2. **矩阵操作实践** (Matrix Operations Practice)：矩阵的创建、运算和性质
3. **实际应用案例** (Real-world Applications)：网络安全中的线性代数应用
4. **性能对比** (Performance Comparison)：NumPy与纯Python的性能差异

## 练习题 (Exercises)

### 练习1：向量运算实现 (Exercise 1: Vector Operations Implementation)
手动实现向量的基本运算（不使用NumPy的高级函数），加深对运算原理的理解。
Manually implement basic vector operations (without NumPy advanced functions) to deepen understanding.

### 练习2：矩阵乘法验证 (Exercise 2: Matrix Multiplication Verification)
实现矩阵乘法算法，并与NumPy结果对比验证正确性。
Implement matrix multiplication algorithm and verify correctness against NumPy results.

### 练习3：网络数据标准化 (Exercise 3: Network Data Normalization)
对给定的网络安全数据进行标准化处理，分析标准化前后的数据分布。
Standardize given network security data and analyze distribution before/after normalization.

### 练习4：相似性度量 (Exercise 4: Similarity Metrics)
实现多种向量相似性度量方法（欧几里得距离、余弦相似度等）。
Implement various vector similarity metrics (Euclidean distance, cosine similarity, etc.).

## 重要概念总结 (Key Concepts Summary)

- **向量** (Vectors)：表示数据点和特征的基本数学结构
- **矩阵** (Matrices)：组织和处理多维数据的工具
- **矩阵乘法** (Matrix Multiplication)：机器学习中最核心的运算
- **标准化** (Normalization)：消除不同特征间量纲差异的重要预处理步骤

## 为什么学习线性代数 (Why Study Linear Algebra)

在网络安全的机器学习应用中，线性代数无处不在：
Linear algebra is ubiquitous in ML applications for cybersecurity:

- 特征向量表示网络连接的属性
- 矩阵运算实现数据变换和模型计算
- 距离度量帮助检测异常行为
- 降维技术减少数据复杂度

## 下节预告 (Next Lesson Preview)

下一课我们将继续学习线性代数，重点关注特征值、特征向量以及它们在数据分析中的应用。
Next lesson we'll continue with linear algebra, focusing on eigenvalues, eigenvectors, and their applications in data analysis.

---

**数学是机器学习的语言，掌握线性代数是成功的第一步！**  
**Mathematics is the language of machine learning - mastering linear algebra is the first step to success!**