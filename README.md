# machine-learning
machine learning note

# 数学

#### 函数，变元

考虑一个函数:
$$
\text { function(input) }
$$

1. $function$ 是标量， 则$function $ 是 $input$ **所有元素所组成的函数**。

例如：$
\boldsymbol{X}_{3 \times 2}=\left(x_{i j}\right)_{i=1, j=1}^{3,2}
$ ，则有：
$$
f(\boldsymbol{X})=a_{1} x_{11}^{2}+a_{2} x_{12}^{2}+a_{3} x_{21}^{2}+a_{4} x_{22}^{2}+a_{5} x_{31}^{2}+a_{6} x_{32}^{2}
$$

2. $function$是向量或者矩阵，$function$ 中的**每个元素**都是 $input$ **中所有元素组成的标量函数**。

例如：$\boldsymbol{X}_{3 \times 2}=\left(x_{i j}\right)_{i=1, j=1}^{3,2}$ ，则：
$$
\begin{aligned}
\boldsymbol{F}_{3 \times 2}(\boldsymbol{X}) &=\left[\begin{array}{ll}
f_{11}(\boldsymbol{X}) & f_{12}(\boldsymbol{X}) \\
f_{21}(\boldsymbol{X}) & f_{22}(\boldsymbol{X}) \\
f_{31}(\boldsymbol{X}) & f_{32}(\boldsymbol{X})
\end{array}\right] \\
&=\left[\begin{array}{cc}
x_{11}+x_{12}+x_{21}+x_{22}+x_{31}+x_{32} & 2 x_{11}+x_{12}+x_{21}+x_{22}+x_{31}+x_{32} \\
3 x_{11}+x_{12}+x_{21}+x_{22}+x_{31}+x_{32} & 4 x_{11}+x_{12}+x_{21}+x_{22}+x_{31}+x_{32} \\
5 x_{11}+x_{12}+x_{21}+x_{22}+x_{31}+x_{32} & 6 x_{11}+x_{12}+x_{21}+x_{22}+x_{31}+x_{32}
\end{array}\right]
\end{aligned}
$$

#### 矩阵求导

$fuction$ 中的每个$f$ 对变元中的每个$x$ 求偏导

分子布局：列向量在分子

分母布局：列向量在分母



# tensorflow2.0













