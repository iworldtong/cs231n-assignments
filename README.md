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
  1. L1(Manhattan) distance ：<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default">$$d_1(\textbf I_1, \textbf I_2) = \sum_p|\textbf I_1^p - \textbf I_2^p|$$</script>
  2. L2(Euclidean)  distance ：$d_2(\textbf I_1, \textbf I_2) = \sqrt{\sum_p(|\textbf I_1^p - \textbf I_2^p|)^2}$
- **说明**
  算法预测准确率比较低，k=1时大概为27%左右。可以全局搜索k的最优值。

### SVM

- **代码实现：assign1_svm.py**

- **多目标代价函数**

  1. hinge loss ：$L_i=\sum_{j\ne y_i}max(0, s_j-s_{y_i}+\Delta)$

  2. square hinge loss ：$L_i=\sum_{j\ne y_i}max(0, s_j-s_{y_i}+\Delta)^2$

     其中，$y_i$指第$i$个样本的真实标签，$s_j$代表输入第$i$个样本后预测为第$j$类的可能性。

- **采用随机梯度下降优化**
  SVM的梯度为：
  $$
  \begin{cases}
  \begin{eqnarray}
  \nabla_{w_{y_i}}L_i &=& -\left(\sum_{k \ne y_i} 1(w_k^Tx_i-w_{y_i}^Tx_i+\Delta \gt 0)  \right)x_i \quad  &j=&y_i\\
  \nabla_{w_{j}}L_i &=& 1(w_j^Tx_i-w_{y_i}^Tx_i+\Delta \gt 0)x_i  &j\ne&y_i
  \end{eqnarray}
  \end{cases}
  $$
  ​

  ​

