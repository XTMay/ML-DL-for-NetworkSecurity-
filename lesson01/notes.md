# 第1课：Python基础与数据结构
# Lesson 1: Python Basics & Data Structures

## 学习目标 (Learning Objectives)

通过本课学习，您将能够：
- 熟练掌握Python基础数据结构的使用 (Master basic Python data structures)
- 理解函数定义和参数传递机制 (Understand function definition and parameter passing)
- 掌握numpy库的基本操作 (Master basic numpy operations)
- 为后续机器学习课程打下坚实基础 (Build solid foundation for machine learning)

## 课程内容 (Course Content)

### 1. Python基础数据结构 (Basic Data Structures)

#### 列表 (Lists)
列表是Python中最常用的数据结构，用于存储有序的元素集合。
Lists are the most commonly used data structure in Python for storing ordered collections of elements.

```python
# 创建列表 (Creating lists)
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]
mixed = [1, "hello", 3.14, True]

# 列表操作 (List operations)
numbers.append(6)        # 添加元素 (Add element)
numbers.insert(0, 0)     # 插入元素 (Insert element)
numbers.remove(3)        # 删除元素 (Remove element)
length = len(numbers)    # 获取长度 (Get length)
```

#### 字典 (Dictionaries)
字典用于存储键值对，提供快速的数据查找。
Dictionaries store key-value pairs for fast data lookup.

```python
# 创建字典 (Creating dictionaries)
student = {
    "name": "Alice",
    "age": 20,
    "major": "Computer Science"
}

# 字典操作 (Dictionary operations)
student["gpa"] = 3.8          # 添加键值对 (Add key-value pair)
age = student.get("age", 0)   # 安全获取值 (Safe get value)
keys = student.keys()         # 获取所有键 (Get all keys)
```

#### 集合 (Sets)
集合用于存储不重复的元素。
Sets store unique elements without duplicates.

```python
# 创建集合 (Creating sets)
unique_numbers = {1, 2, 3, 4, 5}
colors = set(["red", "green", "blue", "red"])  # 自动去重

# 集合操作 (Set operations)
unique_numbers.add(6)                    # 添加元素
intersection = {1, 2, 3} & {2, 3, 4}    # 交集
union = {1, 2, 3} | {3, 4, 5}           # 并集
```

#### 元组 (Tuples)
元组是不可变的有序数据结构。
Tuples are immutable ordered data structures.

```python
# 创建元组 (Creating tuples)
coordinates = (10, 20)
rgb_color = (255, 128, 0)

# 元组解包 (Tuple unpacking)
x, y = coordinates
r, g, b = rgb_color
```

### 2. 函数定义与参数传递 (Function Definition & Parameter Passing)

#### 基础函数定义 (Basic Function Definition)

```python
def greet(name):
    """
    简单的问候函数 (Simple greeting function)
    """
    return f"Hello, {name}!"

# 带默认参数的函数 (Function with default parameters)
def power(base, exponent=2):
    """
    计算幂次方 (Calculate power)
    """
    return base ** exponent

# 可变参数函数 (Variable arguments function)
def sum_all(*args):
    """
    计算所有参数的和 (Sum all arguments)
    """
    return sum(args)
```

#### 高级参数传递 (Advanced Parameter Passing)

```python
# 关键字参数 (Keyword arguments)
def create_profile(name, age, **kwargs):
    profile = {"name": name, "age": age}
    profile.update(kwargs)
    return profile

# 函数作为参数 (Function as parameter)
def apply_operation(numbers, operation):
    return [operation(x) for x in numbers]

# 匿名函数 (Lambda functions)
square = lambda x: x ** 2
numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))
```

### 3. NumPy基础操作 (NumPy Fundamentals)

#### 数组创建与基本操作 (Array Creation and Basic Operations)

```python
import numpy as np

# 创建数组 (Creating arrays)
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.zeros((3, 4))          # 零数组
arr3 = np.ones((2, 3))           # 全一数组
arr4 = np.arange(0, 10, 2)       # 等差数列
arr5 = np.random.random((3, 3))  # 随机数组

# 数组属性 (Array properties)
print(f"形状: {arr1.shape}")      # Shape
print(f"维度: {arr1.ndim}")       # Dimensions
print(f"数据类型: {arr1.dtype}")  # Data type
```

#### 数组运算 (Array Operations)

```python
# 元素级运算 (Element-wise operations)
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

addition = a + b        # 加法
multiplication = a * b  # 乘法
division = a / b        # 除法

# 数学函数 (Mathematical functions)
sqrt_a = np.sqrt(a)
sin_a = np.sin(a)
log_a = np.log(a)

# 统计函数 (Statistical functions)
mean_val = np.mean(a)
std_val = np.std(a)
max_val = np.max(a)
```

### 4. 实用技巧与最佳实践 (Useful Tips & Best Practices)

#### 列表推导式 (List Comprehensions)
```python
# 基础列表推导式 (Basic list comprehension)
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# 字典推导式 (Dictionary comprehension)
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}
```

#### 异常处理 (Exception Handling)
```python
def safe_divide(a, b):
    """
    安全除法函数 (Safe division function)
    """
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("错误：除数不能为零 (Error: Division by zero)")
        return None
    except TypeError:
        print("错误：参数类型不正确 (Error: Invalid argument type)")
        return None
```

## 代码示例讲解 (Code Example Explanation)

本课的 `example.py` 文件包含了以下内容：

1. **数据结构演示** (Data Structure Demo)：展示各种数据结构的创建和操作
2. **函数定义示例** (Function Definition Examples)：不同类型的函数定义
3. **NumPy操作演示** (NumPy Operations Demo)：基础数组操作和数学运算
4. **实际应用场景** (Practical Use Cases)：模拟网络安全数据处理场景

## 练习题 (Exercises)

### 练习1：数据结构操作 (Exercise 1: Data Structure Operations)
创建一个函数，接收一个IP地址列表，返回去重后的IP地址及其出现次数。
Create a function that takes a list of IP addresses and returns unique IPs with their occurrence counts.

### 练习2：数据分析函数 (Exercise 2: Data Analysis Function)
编写一个函数，分析网络日志数据，计算每个用户的访问频率。
Write a function to analyze network log data and calculate access frequency for each user.

### 练习3：NumPy数组处理 (Exercise 3: NumPy Array Processing)
使用NumPy处理网络流量数据，计算统计信息（均值、标准差、最值）。
Use NumPy to process network traffic data and calculate statistics (mean, std, min/max).

### 练习4：综合应用 (Exercise 4: Comprehensive Application)
结合所学知识，创建一个简单的网络安全数据预处理工具。
Combine learned concepts to create a simple network security data preprocessing tool.

## 关键概念总结 (Key Concepts Summary)

- **数据结构选择** (Data Structure Choice)：根据使用场景选择合适的数据结构
- **函数设计** (Function Design)：编写清晰、可重用的函数
- **NumPy高效计算** (NumPy Efficient Computing)：利用NumPy进行高效数值计算
- **代码可读性** (Code Readability)：编写易读、易维护的代码

## 下节预告 (Next Lesson Preview)

下一课我们将学习线性代数基础，包括向量和矩阵的基本概念及其在NumPy中的实现。
Next lesson we'll cover linear algebra fundamentals, including basic concepts of vectors and matrices and their implementation in NumPy.

---

**记住：编程是一门实践的艺术，多动手练习才能真正掌握！**  
**Remember: Programming is a practical art - hands-on practice is key to mastery!**