## papernote
论文笔记

### 主要工作
  + 搜索论文，评估论文价值
  + 下载论文，重命名论文，把论文放到合适目录下
  + 跑预先写好的脚本
    + 生成笔记模板文档
    + 更新README文件
  + 阅读论文
  + 撰写笔记文稿
  + 添加到git，提交，上传

### 结构格式
  + README文件
    + 自动生成README文件
    + 格式：论文发表年月+论文名称+[论文](论文文件相对路径)+[笔记](Markdown文件相对路径)
    + 格式样例：
        ```
        ## [预训练模型](https://github.com/xwzhong/papernote/tree/master/transformer/)
        * _2019.09_ ALBERT: A Lite BERT for Self-supervised Learning of Language Representations [【论文】](https://arxiv.org/pdf/1909.11942.pdf)[【笔记】](https://github.com/xwzhong/papernote/blob/master/transformer/ALBERT%20A%20LITE%20BERT%20FOR%20SELF-SUPERVISED%20LEARNING%20OF%20LANGUAGE%20REPRESENTATIONS.md)
        * _2019.07_ RoBERTa: A Robustly Optimized BERT Pretraining Approach [【论文】](https://arxiv.org/abs/1907.11692)[【笔记】](https://github.com/xwzhong/papernote/blob/master/transformer/RoBERTa_A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)
        * _2019.04_ ERNIE: Enhanced Representation through Knowledge Integration [【论文】](https://arxiv.org/pdf/1904.09223.pdf)[【笔记】](https://github.com/xwzhong/papernote/blob/master/transformer/ERNIE%20Enhanced%20Representation%20through%20Knowledge%20Integration.md)
        * _2018.10_ BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [【论文】](https://arxiv.org/abs/1810.04805)[【笔记】](https://github.com/xwzhong/papernote/blob/master/transformer/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding.md)
        * _2018.06_ Improving Language Understanding by Generative Pre-Training \[[amazonaws](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)[【笔记】](https://github.com/xwzhong/papernote/blob/master/transformer/Improving%20Language%20Understanding%20by%20Generative%20Pre-Training.md)
        * _2018.03_ Universal Sentence Encoder [【论文】](https://arxiv.org/abs/1803.11175)[【笔记】](https://github.com/xwzhong/papernote/blob/master/transformer/Universal%20Sentence%20Encoder.md)
        * _2017.06_ Attention is All You Need [【论文】](https://arxiv.org/abs/1706.03762)[【笔记】](https://github.com/xwzhong/papernote/blob/master/transformer/Attention%20is%20All%20You%20Need.md)
        ```

  + 论文文件
    + 文件名称为Windows环境支持的论文标题名称
      + 把论文名称中的Windows文件名不支持的字符替换为下划线：_
      + 不支持的字符：< > / \ | :  * ? "
      + Windows系统的文件名支持255个英文字符（1个中文字符相当于2个英文字符）
      + 如果论文名称字符数超过200个英文字符，则截断到200个英文字符+...+文件后缀名

  + 笔记文件
    + 文件名称为：论文文件名称+.note.md
    + 笔记文件和论文文件在同一目录下
    + 笔记模块：摘要+笔记+资料
    + 笔记样例：
        ```
        ### 摘要
          1. title
            + BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
            + BERT：语言理解的深度双向变换器预训练
          2. abstract
            + 本文介绍一种称之为BERT的新语言表征模型，意为来自变换器的双向编码器表征量(BidirectionalEncoder Representations from Transformers)。不同于最近的语言表征模型(Peters等，2018; Radford等，2018)，BERT旨在基于所有层的左、右语境来预训练深度双向表征。因此，预训练的BERT表征可以仅用一个额外的输出层进行微调，进而为很多任务(如问答和语言推理)创建当前最优模型，无需对任务特定架构做出大量修改。
            + BERT的概念很简单，但实验效果很强大。它刷新了11个NLP任务的当前最优结果，包括将GLUE基准提升至80.4%(7.6%的绝对改进)、将MultiNLI的准确率提高到86.7%(5.6%的绝对改进)，以及将SQuADv1.1问答测试F1的得分提高至93.2分(1.5分绝对提高)——比人类性能还高出2.0分。  
          3. conclusion
            + 近期实验改进表明，使用迁移学习语言模型展示出的丰富、无监督预训练，是许多语言理解系统的集成部分。特别是，这些结果使得即使低资源任务，也能从很深的单向架构中受益。我们的主要贡献是将这些发现进一步推广到深度双向架构，允许其相同的预训练模型去成功解决一系列广泛的NLP任务。
            + 虽然实验结果很强，在某些情况下超过人类性能，但重要的未来工作是研究BERT能不能捕获其语言现象。
        
        ### 笔记
        #### note:
        &nbsp;&nbsp;&nbsp;&nbsp;paper在论文"Attention Is All You Need"提出的transformer基础上使用双向编码，并基于“大规模语料”+“两项特定的task”预训练参数，最终在task specific任务中fine tuning，得到论文所有呈现数据集中的最优结果。
          1. bidirectional transformer。[OpenAI GPT](https://blog.openai.com/language-unsupervised/)提出了left-to-right transformer，其仅为单向（every token can only attended to previous tokens in the self-attention layers of the Transformer），同时虽然biLSTM虽然也使用了双向结构，但是这两个方向的LSTM相对独立，paper5.1部分详细实验了去除双向结构的影响。各结构对比详见原文fig1。
        
          2. pretrain task。数据使用BooksCorpus (800M words) (Zhu et al., 2015) and English Wikipedia (2,500M words)，同时以下面两种task预训练（最终的loss——the sum of the mean masked LM likelihood and mean next sentence prediction likelihood）：
              + Task #1: Masked LM。以15%随机选取句子的中token，同时对于这15%的词——80%用[mask]代替，10%随机选取一个token，10%保留原始的token，对15%token进行细分的原因:
                 + a mismatch between pre-training and fine- tuning, since the [MASK] token is never seen dur- ing fine-tuning
                 + downside of using an MLM is that only 15% of tokens are predicted in each batch, which suggests that more pre-training steps may be required for the model to converge
              + Task #2: Next Sentence Prediction。由于很多下游任务需要衡量两个句子之间的关系，因此预训练时使用Next Sentence Prediction任务尽量描述这种关系。其中，不同句子部分使用不同的segment embeddings。
              
          3. 实验结果:
            + [GLUE](https://gluebenchmark.com/leaderboard)。11项任务都摘得第一，BERT_large平均score比原最优高6.7;
            + [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)。single模型在f1值上超越人类并超过所有其它学者提出的模型；
            + CoNLL-2003 NER。F1 92.6%(原最优)，BERT_large为92.8%;
            + [SWAG](https://leaderboard.allenai.org/swag/submissions/public)。Dev：BERT_base 81.6%；Test：BERT_large 86.3%，openAI GPT 77.97%；human 88%。
          
          4. paper对比了是否使用bidirectional transformer、Next Sentence Prediction等模块，详见5.1，同时发现（5.2），增大模型大小能明显提高各任务的准确率。
        
        #### highlight:
          1. We also observed that large data sets (e.g., 100k+ labeled training examples) were far less sensitive to hyperparameter choice than small data sets.
          2. Now you have two representations of each word, one left-to-right and one right-to-left, and you can concatenate them together for your downstream task. But intuitively, it would be much better if we could train a single model that was deeply bidirectional(from reddit讨论).
          3. [SWAG (Situations With Adversarial Generations)](https://leaderboard.allenai.org/swag/submissions/public) is a dataset for studying grounded commonsense inference. It consists of 113k multiple choice questions about grounded situations: each question comes from a video caption, with four answer choices about what might happen next in the scene. The correct answer is the (real) video caption for the next event in the video; the three incorrect answers are adversarially generated and human verified, so as to fool machines but not humans.
        
        #### comment：
          1. 文章可谓NLP领域突破性进展，单一模型能达到如此惊人的效果，其主要得益于双向transformer，大语料特定方式的预训练。强烈推荐看more下reddit讨论部分，上面有原作一些回复；
          2. 训练相同参数下的BERT_large在8 P100s下训练要一年？（详见reddit讨论，注：论文max len为512）；
          3. 针对SWAG数据，它的形式比较像对话系统，将该算法用于闲聊是不是可以有较大的提升（对于多轮编码，可以使用多个segment embeddings）；
          4. pretrain已证明了在各项任务中的优越性，但是针对具体的任务，预训练什么模型结构，用什么样的数据仍需考究。
        
        ### 资料
          1. [reddit讨论](https://www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/)
          2. [全面超越人类！Google称霸SQuAD，BERT横扫11大NLP测试](http://www.qianjia.com/html/2018-10/13_307585.html)
          3. [code](https://github.com/google-research/bert)
          4. [深入理解BERT Transformer ，不仅仅是注意力机制](https://www.jiqizhixin.com/articles/2019-03-19-13)
          5. [Bert时代的创新：Bert在NLP各领域的应用进展](https://www.jiqizhixin.com/articles/2019-06-10-7)
          6. [【中文版 | 论文原文】BERT：语言理解的深度双向变换器预训练](https://www.cnblogs.com/guoyaohua/p/bert.html)
        ```  
