# cs231n-assignments
斯坦福cs231n作业实践

### [课程主页](http://cs231n.github.io/)

### [cifar10数据集下载](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

`python download_cifar10.py` 或 `wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`

---

## [Assignment 1](http://cs231n.github.io/assignments2016/assignment1/)

### knn

- **代码实现：assign1_knn.py**
- **两种距离度量方法**
  1. L1(Manhattan) distance ：$d\_1(\textbf I\_1, \textbf I\_2) = \sum\_p|\textbf I\_1^p - \textbf I\_2^p|$ 
  2. L2(Euclidean)  distance ：$d\_2(\textbf I\_1, \textbf I\_2) = \sqrt{\sum\_p(|\textbf I\_1^p - \textbf I\_2^p|)^2}$

### SVM

- **代码实现：assign1_svm.py**

- **多目标代价函数**

  1. hinge loss ：$L\_i=\sum\_{j\ne y\_i}max(0, s\_j-s\_{y\_i}+1)$
  2. square hinge loss ：$L\_i=\sum\_{j\ne y\_i}max(0, s\_j-s\_{y\_i}+1)^2$

  ​

  ​