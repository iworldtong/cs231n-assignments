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
  1. L1(Manhattan) distance ：$d_1(\textbf I_1, \textbf I_2) = \sum_p|\textbf I_1^p - \textbf I_2^p|$
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



### Softmax

* **代码实现：assign1_softmax.py**

* **代价函数相对于SVM的只是从hinge loss变为cross-entropy loss，即交叉熵损失**
  $$
  L_i= -log(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}})
  $$

* **计算梯度**
  $$
  \begin{cases}
  \begin{eqnarray}
  \nabla_{w_{y_i}} L_i &=& -x_i + \frac{e^{s_j}}{\sum_k e^{s_k}} x_i \quad &j =& y_i  \\
  \nabla_{w_j}L_i &=& \frac{e^{s_j}}{\sum_k e^{s_k}} x_i \quad &j \ne& y_i 
  \end{eqnarray}
  \end{cases}
  $$

### Two Layer Net

* **代码实现：assign1_two_layer_net.py**

* **前向计算 (网络结构)**

  * input $z_0$

  * calc 
    $$
    \begin{eqnarray}
    z_1 &=& z_0w_1 + b_1 \\
    a_1 &=& max(0, z_1)\\
    z_2 &=& a_1w_2+b_2\\
    a_2 &=& softmax(z_2)
    \end{eqnarray}
    $$
    其中，$z_0 \in \mathbb R^{N \times D}，z_1、a_1 \in \mathbb R^{D \times H}， z_2、a_2 \in \mathbb R^{H \times C}$

  * output $argmax(a_2)$

    ​

* **反向传播**

  * 梯度计算
    输出层 (第二层) 的激活函数为$softmax$函数，其梯度公式可参见前文。可以简化的一点是：假定每个样本的标签向量为$one-hot$形式 (只有真实类别索引上的值为1，其余为0)，则可以把之前分开的两个公式写成一个：
    $$
    \begin{eqnarray}
    \nabla_{w_{2_j}} L_i &=& \left( \frac{e^{s_j}}{\sum_k e^{s_k}} - \textbf 1_{y_ij} \right)a_{1_i}\\
    \nabla_{b_{2_j}} L_i &=& \left( \frac{e^{s_j}}{\sum_k e^{s_k}} - \textbf 1_{y_ij} \right)
    \end{eqnarray}
    $$
    其中，$ \begin{cases} \begin{eqnarray} \textbf 1_{y_ij} &=& 0 \quad &j \ne & y_i \\ \textbf1_{y_i y_i} &=& 1 &j =&y_i \end{eqnarray} \end{cases} $.

    隐藏层 (第一层) 的激活函数是$relu$函数，其梯度为所有大于零的输入的梯度值为1。则对每个输入样本$z_{0_i}$，得如下梯度：
    $$
    \begin{eqnarray}
    \frac{da_1}{dw_1} &=& \frac{da_1}{dz_1} \frac{dz_1}{dw_1} \\
    &=& sign(z_1) \cdot z_{0_i}
    \end{eqnarray}
    $$
    注意上式点乘是一个广播操作，$sign(z_1)$是$H$维向量、$z_{0_i}$是$D$维向量，相乘得到$D \times H$ 大小的矩阵。

    ​
    $$

    $$
