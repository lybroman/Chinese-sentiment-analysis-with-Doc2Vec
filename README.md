# Chinese-sentiment-analysis-with-Doc2Vec


----------

## 简介

中文语料的情感分析基本步骤如下：

 - 爬取相关的语料或者下载相关语料（本文使用了对于宾馆评价的相关语料作为例子）
 - 将语料进行预处理并分词
 - 用某种量化的表达形式来对语料进行数字化处理
 - 基于监督学习的分类器训练

开发环境`Python-v3(3.6)`：
        gensim==3.0.1
        jieba==0.39
        scikit-learn==0.19.1
        tensorflow==1.2.1
        numpy==1.13.1+mkl

示例代码参考[Chinese-sentiment-analysis-with-Doc2Vec][1]


在repo中有两个zip文件分别为`train.zip`和`test.zip`数据，当然你也可以直接在加载语料时将部分数据用作测试数据（详见后文）。

## 数据预处理(`preprocess.py`)

 - zip数据中为大量的txt文档，每一个的后缀是评分，例如`72_1380108_2006-11-9_1.0.txt`，那么该评分为1.0分（其实就是差评啦）。我们需要做的是将所有评分划分为1、2、3、4，5档，顾名思义就是评价由坏到好。这里用了一些简单的字符串处理来获取分数并使用`round`函数来对分数取整。
 - 将不同的评分txt按folder分类放好

## 分词（`words_segment.py`）

 - 分词是通过第三方的[jieba][2]实现的。
 - 分词之前需要做一些简单的处理，比如过滤一些不感兴趣的字符。

```
    filter_chars = "\r\n，。；！,.:;：、"
    trans_dict = dict.fromkeys((ord(_) for _ in filter_chars), '')
    line = line.translate(trans_dict)
```
 - 将分完词的语料按照分数归并到同一个文本做为训练做准备

## 文本向量化模型（`main.py：step 1-3`）

- 这里使用到了`gensim.models.doc2vec`，该模块提供了将不定长的文本映射到维度大小固定的向量的功能。这对于计算相似度还是用作后续的CNN分类器训练（后续有时间的话会实现基于TensorFlow的分类器）都是十分有帮助的。
- 具体的原理可以参考[distributed-representations-of-sentences-and-documents][3]
- gensim [doc2vec][4]
- 本文旨在通过简单的示例介绍如何通过训练模型来自动判断某个新的输入评价是好评（5分）还是差评（1分），所以在后续的代码中，使用的样本就来自于这两类样本的集合（后续有时间的话会继续实现多分类问题）

## 训练分类器（`main.py：step 4-5`）

- 这里使用了`sklearn`中的分类器（LR、SVM、决策树等等，最新版本的sklearn还提供了NN的实现）。具体参考[scikit-learn][5]。
- 数据的标记十分简单，将5分的训练集标记为1，1分的训练集标记为0即可（如果实现多分类，按照分数标记即可。）
- 其中我把20%的训练集抽出作为测试数据:

```
    train, test, train_label, test_label = ms.train_test_split(
        train_arrays, train_labels, test_size=0.2)
```
- 最后进行验证，一般>0.6就认为是一个有不错预测能力的模型了

## 新样本预测（`prediction.py`）
- 通过加载之前训练的model和分类器对测试样本进行预测
- 同时记录了每一个测试样本最近似的训练样本

## 后续工作
- 实现多分类
- 基于TF的CNN分类器

  [1]: https://github.com/lybroman/Chinese-sentiment-analysis-with-Doc2Vec
  [2]: https://github.com/fxsjy/jieba
  [3]: https://blog.acolyer.org/2016/06/01/distributed-representations-of-sentences-and-documents/
  [4]: https://radimrehurek.com/gensim/models/doc2vec.html
  [5]: http://scikit-learn.org/stable/
