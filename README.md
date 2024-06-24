随着物联网的普及，越来越多的设备被引入市场，导致安全漏洞日益增多。在这种环境下，物联网设备识别方法作为预防性安全措施变得非常重要。本文研究并改进了一种基于深度学习物联网设备识别算法。我们对物联网设备的网络包数据进行了预处理，将有效负载部分转换为固定大小的伪图像。与传统方案不同，我们采用了二维卷积（2D Convolution）来提取伪图像中的空间特征，并通过全局平均池化（Global Average Pooling）代替全连接层。提取的空间特征输入到LSTM中，捕捉时间序列的依赖关系，进一步丰富特征表示。实验结果表明，我们的改进模型在准确率、召回率、精确率和F1评分等多个评估指标上均优于传统的单一模型以及文献中的其他方案
