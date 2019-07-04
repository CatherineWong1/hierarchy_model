# two_level_sim
整体模型设计思路：
1. Data Preprocess
所针对的数据：
1. 标题内容
1.1 标题内容
1.1.1 标题内容
1.1.2 标题内容
1.2 标题内容
1.3 .......
数据预处理之数据集：
src.txt格式：
文字行：一个标题下面的所有内容（包括分段的内容）
一行空行：空行出现代表一个大标题的结束
两行空行：
tgt.txt格式：
一行：和src.txt中的文字行对应，真正的内容是文字的内容
一行空行：和src.txt中的空行对应
两行空行：和src.txt中的空行对应

备注：之所以加两个空行的原因是为了以后思路有变时，处理数据方便

数据预处理之Vocabulary
可以参考opennmt中或者其他的生成Vocabulary的函数，因此初始化模型参数时，要有vocabulary的size大小


2. Document Embedding
前提：Document代表一个完整标题及其下面所有子标题的内容
将一行空行前的所有文字行取出，对每一行进行Embedding。
利用Bert进一步进行Document的embedding,在送入Bert中得到Embedding时，需要进行Mode的判断
根据Mode，选择BertModel的函数。

3. Model
Encoder的部分：
将Embedding送入Bert,重新得到的向量可视为Encoder出来的向量。
假设有N行文字，则需要得到N个Bert张量，对于当前张量，将当前张量和其余向量的均值concatenate到一起。
重复N次，重新获得N个张量，随后将张量送入Decoder中。

Decoder部分：
送入一层LSTM，128个Unit即可。得到最终的结果

softmax
Pvoc = softmax(wx+b)，x即Decoder output出来的张量，w,b都是需要学习的参数
由于是生成式的，而且长度有限制，也不需要逐个step去生成，因此直接选择top k个distribution来组成新的标题。


loss
使用cross entropy
方案一：
对生成的一级标题和gold standard的标题进行cross entropy

方案二：
求出N个向量对应的标题及其gold standard的标题，并进行cross entropy
最后得到一个均值作为loss

optimizer
Adagrad


