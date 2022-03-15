# Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding

WSDM2018

### 1 论文概述

#### 1.1 Abstract

Caser是一个序列推荐模型，结合了用户的长期偏好以及短期兴趣。**Caser强调的是短期信息的影响。**

#### 1.2 **序列模式的定义**

![image-20220303163039039](C:\Users\97399\AppData\Roaming\Typora\typora-user-images\image-20220303163039039.png)

文章针对基于马尔可夫链的序列推荐方法的限制性，总结出三种序列模式：

- point-level：基于马尔可夫链的模型便是这种序列模式，蓝色的序列行为都只是单独的影响下一个行为；
- union-level，no skip：(b)中就是一种联合的序列模式，同时考虑三个蓝色行为对下一个行为的影响；
- union-level，skip once：作者还考虑了一种跳跃的行为。

### 2. **模型结构**

![image-20220303163309550](C:\Users\97399\AppData\Roaming\Typora\typora-user-images\image-20220303163309550.png)

整个模型包含三个部分：

- Embedding层对用户、物品序列进行密集型表示，用户的表示可以理解为用户的 general preference；
- 卷积层（CNN）学习用户短期（![[公式]](https://www.zhihu.com/equation?tex=L)时间内）的序列特征；
- 全连接层将 拼接的序列特征与用户偏好 映射到用户在当前时间与每个物品交互的可能性；

#### 2.1 Embedding

在物品池中，定义用户历史行为中有前![[公式]](https://www.zhihu.com/equation?tex=L)个物品，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BQ%7D_i+%5Cin+%5Cmathbb%7BR%7D%5Ed)为物品![[公式]](https://www.zhihu.com/equation?tex=i)的embedding信息。对于用户![[公式]](https://www.zhihu.com/equation?tex=u)，在![[公式]](https://www.zhihu.com/equation?tex=t)时刻，前![[公式]](https://www.zhihu.com/equation?tex=L)个物品的embedding信息进行堆叠，得到一个embedding矩阵![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BE%7D%5E%7B%28u%2C+t%29%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7BL+%5Ctimes+d%7D)：

![[公式]](https://www.zhihu.com/equation?tex=E%5E%7B%28u%2C+t%29%7D%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7DQ+s_%7Bt-L%7D%5E%7Bu%7D+%5C%5C+%5Cvdots+%5C%5C+Q_%7BS_%7Bt-2%7D%5E%7Bu%7D%7D+%5C%5C+Q_%7BS_%7Bt-1%7D%5E%7Bu-1%7D%7D%5Cend%7Barray%7D%5Cright%5D+%5C%5C)

对于用户![[公式]](https://www.zhihu.com/equation?tex=u)，embedding信息为![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BP%7D_u+%5Cin+%5Cmathbb%7BR%7D%5Ed)。

作者主要是通过卷积层来提取用户的短期兴趣，因此![[公式]](https://www.zhihu.com/equation?tex=L)的选择是人为的超参数，在实验中，![[公式]](https://www.zhihu.com/equation?tex=L)的取值为1～9；

#### 2.2 **卷积层**

作者将embedding矩阵![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BE%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7BL+%5Ctimes+d%7D)看作“图片”，即CNN的输入。这样就可以使用CNN中的过滤器（filter）来搜索局部的序列模式。

对于卷积层的过滤器，文章分为两种方式，**水平过滤器（horizontal filters）和垂直过滤器（vertical filter）**，来提取不同的序列模式信息。

##### 2.2.1 水平卷积层

对于水平卷积层通过多个联合的物品信息来捕获一个**union-level**，作者给出了一个可视化展示，如下图所示。

![image-20220303163744841](C:\Users\97399\AppData\Roaming\Typora\typora-user-images\image-20220303163744841.png)

首先采用![[公式]](https://www.zhihu.com/equation?tex=n)个水平过滤器![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BF%7D%5E%7Bk%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7Bh+%5Ctimes+d%7D%2C+1+%5Cleq+k+%5Cleq+n+) ， ![[公式]](https://www.zhihu.com/equation?tex=h+%5Cin%5C%7B1%2C+%5Ccdots%2C+L%5C%7D)是过滤器的高，**宽必须为![[公式]](https://www.zhihu.com/equation?tex=d)**。然后过滤器在![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BE%7D)上滑动，进行![[公式]](https://www.zhihu.com/equation?tex=i%281+%5Cleq+i+%5Cleq+L+-+h+%2B1+%29+)次的水平维度交互。故第![[公式]](https://www.zhihu.com/equation?tex=i)次通过卷积得到的值为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bc%7D_%7Bi%7D%5E%7Bk%7D%3D%5Cphi_%7Bc%7D%5Cleft%28%5Cmathbf%7BE%7D_%7Bi%3A+i%2Bh-1%7D+%5Codot+%5Cmathbf%7BF%7D%5E%7Bk%7D%5Cright%29+%5C%5C)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Codot)为内积操作，![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%28%5Ccdot%29)为激活函数。因此全部通过![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BF%7D%5E%7Bk%7D)进行卷积得到的向量为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bc%7D%5E%7Bk%7D%3D%5Cleft%5B%5Cmathbf%7Bc%7D_%7B1%7D%5E%7Bk%7D+%5Cmathbf%7Bc%7D_%7B2%7D%5E%7Bk%7D+%5Ccdots+%5Cmathbf%7Bc%7D_%7BL-h%2B1%7D%5E%7Bk%7D%5Cright%5D+%5C%5C)

卷积操作过后，通过**最大池化操作（max pooling）**来捕获过滤器提取的最重要的特征。对于![[公式]](https://www.zhihu.com/equation?tex=n)个过滤器，最终输出![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bo%7D+%5Cin+%5Cmathbb%7BR%7D%5En)为

![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7Bo%7D%3D%5Cleft%5C%7B%5Cmax+%5Cleft%28%5Cboldsymbol%7Bc%7D%5E%7B1%7D%5Cright%29%2C+%5Cmax+%5Cleft%28%5Cboldsymbol%7Bc%7D%5E%7B2%7D%5Cright%29%2C+%5Ccdots%2C+%5Cmax+%5Cleft%28%5Cboldsymbol%7Bc%7D%5E%7Bn%7D%5Cright%29%5Cright%5C%7D+%5C%5C)

##### 2.2.1 垂直卷积层

选择![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bn%7D)个![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7BF%7D%7D%5E%7Bk%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7BL+%5Ctimes+1%7D%2C+1+%5Cleq+k+%5Cleq+%5Ctilde%7Bn%7D)过滤器， ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BE%7D) 经过垂直卷积操作，得到结果![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D%5Ek+%5Cin+%5Cmathbb%7BR%7D%5Ed)：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D%5E%7Bk%7D%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7D%5Ctilde%7Bc%7D_%7B1%7D%5E%7Bk%7D+%5Ctilde%7Bc%7D_%7B2%7D%5E%7Bk%7D+%5Ccdots+%5Ctilde%7Bc%7D_%7Bd%7D%5E%7Bk%7D%5Cend%7Barray%7D%5Cright%5D+%5C%5C)

作者提到上述结果与【![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BE%7D)和![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7BF%7D%5Ek%7D)的权重和】相等：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D%5E%7Bk%7D%3D%5Csum_%7Bl%3D1%7D%5E%7BL%7D+%5Ctilde%7BF%7D_%7Bl%7D%5E%7Bk%7D+%5Ccdot+E_%7Bl%7D+%5C%5C)

因此垂直过滤器![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7BF%7D%7D%5E%7Bk%7D)可以聚合前![[公式]](https://www.zhihu.com/equation?tex=L)个物品的embedding信息，不同的过滤器可以组成不同的聚合，来捕获point-level序列模式信息。

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bn%7D)个垂直过滤器最终得到输出![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7Bo%7D%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7Bd%5Ctilde%7Bn%7D%7D)：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cboldsymbol%7Bo%7D%7D%3D%5Cleft%5B%5Ctilde%7B%5Cboldsymbol%7Bc%7D%7D%5E%7B1%7D+%5Ctilde%7B%5Cboldsymbol%7Bc%7D%7D%5E%7B2%7D+%5Ccdots+%5Ctilde%7B%5Cboldsymbol%7Bc%7D%7D%5E%7B%5Ctilde%7Bn%7D%7D%5Cright%5D+%5C%5C)

与水平过滤器不同的是，垂直过滤器的大小必须是![[公式]](https://www.zhihu.com/equation?tex=L+%5Ctimes+1)，且无需最大池化操作，**因为作者希望在每个维度上保持聚合**。

##### 2.2.1 全连接层

水平卷积层与垂直卷积层得到的![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bo%7D)和![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7Bo%7D%7D)进行拼接，然后通过一个全连接层得到更高级别的特征：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D%3D%5Cphi_%7Ba%7D%5Cleft%28%5Cboldsymbol%7BW%7D%5Cleft%5B%5Cbegin%7Barray%7D%7Bl%7D%5Cboldsymbol%7Bo%7D+%5C%5C+%5Ctilde%7B%5Cboldsymbol%7Bo%7D%7D%5Cend%7Barray%7D%5Cright%5D%2B%5Cboldsymbol%7Bb%7D%5Cright%29+%5C%5C)

为了捕获用户的通用偏好，作者通过embedding，得到了用户信息表示![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BP%7D_u)，且与上述![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D)进行拼接，经过全连接：

![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7By%7D%5E%7B%28u%2C+t%29%7D%3D%5Cboldsymbol%7BW%7D%5E%7B%5Cprime%7D%5Cleft%5B%5Cbegin%7Barray%7D%7Bc%7Dz+%5C%5C+%5Cboldsymbol%7BP%7D_%7Bu%7D%5Cend%7Barray%7D%5Cright%5D%2B%5Cboldsymbol%7Bb%7D%5E%7B%5Cprime%7D+%5C%5C)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BW%7D%27+%5Cin+%5Cmathbb%7BR%7D%5E%7B%7CI%7C%5Ctimes+2d%7D)。
