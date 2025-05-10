# transformer-from-scratch

A step-by-step reimplementation of the original Transformer model in PyTorch, built from scratch for learning purposes.  
Based on the paper [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762).

## Background

​	在 Transformer 出现之前，深度学习领域主要依赖两种主流架构：卷积神经网络（CNN）和循环神经网络（RNN）。其中，CNN 通常应用于图像相关任务，例如图像分类（如 ImageNet 数据集上的分类任务），而 RNN 及其变种（如 LSTM、GRU）则广泛用于建模序列数据，尤其是在自然语言处理任务中，如机器翻译（如基于 RNN 的 Seq2Seq 模型在 WMT 任务中的应用）。然而，尽管这两类模型在各自领域取得了显著成果，它们在`处理长距离依赖`、`并行计算效率`和`建模能力`方面存在一定的局限性。这些挑战为 Transformer 架构的提出奠定了背景与动因。

- 基于卷积神经网络的架构（CNN）

[待补充]

- 基于循环神经网络的架构（RNN）

[待补充]



相较于卷积神经网络（CNN），Transformer 借助自注意力机制（Self-Attention）在每一层直接建模任意两个位置之间的依赖关系，从而具备强大的全局建模能力与良好的可扩展性。虽然 CNN 能通过局部感受野与堆叠层级实现一定程度的长距离建模，但其感受野增长缓慢，难以高效捕捉全局依赖关系。与传统的循环神经网络（RNN）相比，Transformer 完全摒弃了序列化的递归结构，转而通过位置编码（Positional Encoding）引入位置信息，使得模型能够在保持顺序感的同时并行处理整个序列，显著提升了训练效率。此外，Transformer 的核心模块还包括多头注意力机制、前馈神经网络、残差连接以及层归一化，使其在深度堆叠的同时依然保持稳定的训练性能。凭借上述优势，Transformer 架构已逐步取代传统的 RNN 和 CNN，成为自然语言处理、时间序列建模乃至图像理解等多个领域的主流选择。[待补充计算复杂度计算]


### Part01: Model Architecture
**Encoder-Decoder 架构**是一种广泛应用于序列建模任务的神经网络结构，主要包括两个部分：编码器（Encoder）和解码器（Decoder）。
- **编码器**负责接收输入序列，并将其逐步转换为一个固定长度或可变长度的上下文表示（context representation），即高维语义向量。
- **解码器**则基于该上下文表示，逐步生成目标输出序列，通常采用自回归方式（autoregressive），即一步步预测后续输出。
该架构最早应用于神经机器翻译（Neural Machine Translation, NMT），但由于其良好的通用性和表达能力，已经广泛应用于**文本生成、语音识别、图像字幕生成、时间序列预测**等各类输入输出成对的序列任务中。一个标准的Encoder-Decoder架构代码如下：
```python

1
```
