# 机器学习与深度学习在网络安全分析中的应用
# Machine Learning and Deep Learning Applications in Network Security Analysis

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## 课程简介 (Course Introduction)

本课程旨在为具有编程基础但缺乏AI背景的学生提供一个系统性的学习路径，将机器学习和深度学习技术应用于网络安全分析。通过20节循序渐进的课程，学生将掌握从基础数学知识到高级深度学习技术的完整知识体系，并能够在实际的网络安全场景中应用这些技术。

This course aims to provide students with programming background but without AI experience a systematic learning path to apply machine learning and deep learning techniques in network security analysis. Through 20 progressive lessons, students will master the complete knowledge system from basic mathematical concepts to advanced deep learning techniques, and be able to apply these technologies in real network security scenarios.

### 课程目标 (Learning Objectives)

- 掌握机器学习和深度学习的核心概念和算法 (Master core concepts and algorithms of ML/DL)
- 学会使用Python进行数据处理、特征工程和模型训练 (Learn data processing, feature engineering, and model training with Python)
- 理解网络安全中的异常检测、入侵检测等应用场景 (Understand anomaly detection, intrusion detection in cybersecurity)
- 能够独立完成一个完整的网络安全AI项目 (Complete an independent cybersecurity AI project)

### 适合人群 (Target Audience)

- 具有Python编程基础的学生 (Students with Python programming background)
- 对网络安全和人工智能感兴趣的初学者 (Beginners interested in cybersecurity and AI)
- 希望将AI技术应用于安全领域的开发者 (Developers wanting to apply AI in security)

## 学习前提条件 (Prerequisites)

- Python编程基础 (Basic Python programming skills)
- 基本的数学知识（高中水平） (Basic mathematics knowledge - high school level)
- 对网络安全概念有基本了解（推荐但非必需） (Basic understanding of cybersecurity concepts - recommended but not required)

## 快速开始 (Quick Start)

### 1. 安装依赖 (Install Dependencies)

```bash
# 克隆项目 (Clone the repository)
git clone https://github.com/yourusername/ML-DL-for-NetworkSecurity
cd ML-DL-for-NetworkSecurity

# 创建虚拟环境 (Create virtual environment)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖包 (Install dependencies)
pip install -r requirements.txt
```

### 2. 运行示例代码 (Run Example Code)

```bash
# 进入第一课文件夹 (Enter lesson01 folder)
cd lesson01

# 查看课程笔记 (View lesson notes)
cat notes.md

# 运行示例代码 (Run example code)
python example.py

# 完成练习 (Complete exercises)
python exercise.py
```

## 完整课程目录 (Complete Course Catalog)

### 第一部分：基础知识 (Part I: Fundamentals)

#### 第1课：Python基础与数据结构 (Lesson 1: Python Basics & Data Structures)
**学习内容**: 列表、字典、集合、元组、常用操作、函数定义、参数传递、标准库介绍、numpy基础  
**实践项目**: Leetcode简单题 & 编写函数，熟练numpy数组操作  
**文件**: `lesson01/`

#### 第2课：线性代数复习 I (Lesson 2: Linear Algebra I)
**学习内容**: 向量、矩阵基础、矩阵乘法  
**实践项目**: 用numpy实现矩阵乘法  
**文件**: `lesson02/`

#### 第3课：线性代数复习 II (Lesson 3: Linear Algebra II)
**学习内容**: 特征值、特征向量简介  
**实践项目**: 计算小矩阵的特征值和特征向量  
**文件**: `lesson03/`

#### 第4课：概率统计复习 (Lesson 4: Probability & Statistics Review)
**学习内容**: 概率基本概念、随机变量、分布（正态分布）  
**实践项目**: 用numpy模拟正态分布数据  
**文件**: `lesson04/`

#### 第5课：Python数据处理与可视化 (Lesson 5: Data Processing & Visualization)
**学习内容**: pandas基础，matplotlib/seaborn简单绘图  
**实践项目**: 处理CSV数据并绘图  
**文件**: `lesson05/`

### 第二部分：机器学习基础 (Part II: Machine Learning Fundamentals)

#### 第6课：机器学习概念入门 (Lesson 6: Intro to Machine Learning)
**学习内容**: 监督/无监督学习，常见算法概览  
**实践项目**: 数据集划分，手写线性回归  
**文件**: `lesson06/`

#### 第7课：线性回归实战 (Lesson 7: Linear Regression in Practice)
**学习内容**: 理论讲解 + sklearn实现  
**实践项目**: 用真实数据训练回归模型  
**文件**: `lesson07/`

#### 第8课：逻辑回归与分类 (Lesson 8: Logistic Regression & Classification)
**学习内容**: 原理与二分类问题  
**实践项目**: sklearn实现逻辑回归  
**文件**: `lesson08/`

#### 第9课：异常检测基础 (Lesson 9: Anomaly Detection Basics)
**学习内容**: 异常检测概念，常用方法介绍（基于统计、距离）  
**实践项目**: Isolation Forest检测异常  
**文件**: `lesson09/`

#### 第10课：特征工程与数据预处理 (Lesson 10: Feature Engineering & Data Preprocessing)
**学习内容**: 数据清洗、缺失值处理、特征选择、归一化与标准化  
**实践项目**: 对安全数据集进行预处理并生成特征  
**文件**: `lesson10/`

### 第三部分：深度学习基础 (Part III: Deep Learning Fundamentals)

#### 第11课：深度学习基础 (Lesson 11: Intro to Deep Learning)
**学习内容**: 神经网络结构、激活函数  
**实践项目**: numpy手写前馈神经网络  
**文件**: `lesson11/`

#### 第12课：TensorFlow/PyTorch入门 (Lesson 12: TensorFlow/PyTorch Basics)
**学习内容**: 张量操作，简单模型搭建  
**实践项目**: 训练一个简单神经网络  
**文件**: `lesson12/`

#### 第13课：卷积神经网络 (Lesson 13: CNN Basics)
**学习内容**: 卷积层、池化层介绍  
**实践项目**: CNN图像分类  
**文件**: `lesson13/`

#### 第14课：RNN与序列数据 (Lesson 14: RNN & Sequential Data)
**学习内容**: RNN基础，序列模型简介  
**实践项目**: RNN序列分类  
**文件**: `lesson14/`

### 第四部分：网络安全应用 (Part IV: Cybersecurity Applications)

#### 第15课：AI在安全中的应用案例1 (Lesson 15: AI in Security Case Study 1)
**学习内容**: 网络流量异常检测、入侵检测系统  
**实践项目**: 阅读一篇相关论文，写总结  
**文件**: `lesson15/`

#### 第16课：AI在安全中的应用案例2 (Lesson 16: AI in Security Case Study 2)
**学习内容**: 恶意软件/病毒检测  
**实践项目**: 实现基于特征的简单病毒检测模型  
**文件**: `lesson16/`

#### 第17课：深度学习高级技巧 (Lesson 17: Advanced Deep Learning Techniques)
**学习内容**: Dropout、正则化、Batch Normalization、超参数调优  
**实践项目**: 在安全数据集上调优模型性能  
**文件**: `lesson17/`

### 第五部分：项目实战 (Part V: Final Project)

#### 第18课：实验室项目准备 (Lesson 18: Project Preparation)
**学习内容**: 项目需求解析，数据预处理与特征工程  
**实践项目**: 准备并清理一个安全相关数据集  
**文件**: `lesson18/`

#### 第19课：项目实战 (Lesson 19: Final Project)
**学习内容**: 基于前面学习内容，设计并训练检测模型  
**实践项目**: 开发一个简单异常检测demo  
**文件**: `lesson19/`

#### 第20课：课程总结与后续学习规划 (Lesson 20: Course Summary & Next Steps)
**学习内容**: 复习、答疑、分享后续学习资源  
**实践项目**: 制定下一步深造计划  
**文件**: `lesson20/`

## 项目结构 (Project Structure)

```
ML-DL-for-NetworkSecurity/
├── README.md                    # 项目说明文档
├── requirements.txt            # Python依赖包列表
├── data/                       # 示例数据文件夹
│   ├── sample_network.csv     # 网络流量示例数据
│   ├── sample_logs.csv        # 日志示例数据
│   └── malware_features.csv   # 恶意软件特征示例数据
├── lesson01/                   # 第1课：Python基础
│   ├── notes.md               # 课程笔记（中英文对照）
│   ├── example.py             # 示例代码
│   └── exercise.py            # 练习代码框架
├── lesson02/                   # 第2课：线性代数I
│   ├── notes.md
│   ├── example.py
│   └── exercise.py
├── ...                        # 其他课程文件夹
└── lesson20/                   # 第20课：课程总结
    ├── notes.md
    ├── example.py
    └── exercise.py
```

## 推荐学习路径 (Recommended Learning Path)

### 初学者路径 (Beginner Path)
1. **第1-5课**: 扎实基础，重点练习Python和数学知识
2. **第6-10课**: 理解机器学习核心概念，多做练习题
3. **第11-14课**: 循序渐进学习深度学习，重点理解原理
4. **第15-17课**: 学习安全应用，阅读相关论文
5. **第18-20课**: 完成项目实战，巩固所学知识

### 有经验学习者路径 (Experienced Learner Path)
1. **快速浏览第1-5课**: 复习基础概念
2. **重点学习第6-14课**: 深入理解算法原理和实现
3. **专注第15-20课**: 安全应用和项目实战

## 学习建议 (Study Tips)

1. **理论与实践结合** (Combine Theory with Practice): 每课都有对应的代码实践，务必动手编写
2. **循序渐进** (Step by Step): 按顺序学习，确保前置知识掌握牢固
3. **多做练习** (Practice More): 完成所有练习题，巩固知识点
4. **阅读文档** (Read Documentation): 学会查阅官方文档和API说明
5. **参与讨论** (Join Discussions): 在Issues中提问和讨论问题

## 技术栈 (Technology Stack)

- **编程语言**: Python 3.8+
- **数据处理**: NumPy, Pandas, Matplotlib, Seaborn
- **机器学习**: Scikit-learn
- **深度学习**: TensorFlow 2.x / PyTorch
- **数据可视化**: Matplotlib, Seaborn, Plotly
- **开发环境**: Jupyter Notebook, VS Code

## 贡献指南 (Contributing)

欢迎提交问题和改进建议！请遵循以下步骤：

1. Fork此仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

We welcome issues and improvement suggestions! Please follow these steps:
1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 许可证 (License)

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 联系方式 (Contact)

如有问题或建议，请通过以下方式联系：
- 提交Issue: [GitHub Issues](https://github.com/yourusername/ML-DL-for-NetworkSecurity/issues)
- 邮箱: your.email@example.com

For questions or suggestions, please contact us via:
- Submit Issue: [GitHub Issues](https://github.com/yourusername/ML-DL-for-NetworkSecurity/issues)
- Email: your.email@example.com

## 致谢 (Acknowledgments)

- 感谢所有开源库的贡献者
- 感谢网络安全和机器学习社区的支持
- 特别感谢提供数据集和案例研究的研究人员

Thanks to all contributors of open source libraries, the support from cybersecurity and machine learning communities, and especially to researchers who provided datasets and case studies.

---

**开始您的网络安全AI学习之旅吧！🚀**  
**Start your cybersecurity AI learning journey! 🚀** 
