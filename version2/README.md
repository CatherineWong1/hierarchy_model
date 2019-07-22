# 方案二
## 模型所需数据
模型所需的数据格式为：
```
{
	"segment": [{
			"heading": [{
					"src_txt": 原文段落1,
					"tgt_txt": 标题1,
					"src": 原文对应的wordPiece id,
					"segs": 原文对应的句子分隔,
					"tgt": 标题对应的wordPiece id
				},
				{
					"src_txt": 原文段落2,
					"tgt_txt": "",
					"src": 原文对应的wordPiece id,
					"segs": 原文对应的句子分隔,
					"tgt": 标题对应的wordPiece id
				}
			]
		},
		{
			"heading": [{
					"src_txt": 原文段落1,
					"tgt_txt": 标题1.1,
					"src": 原文对应的wordPiece id,
					"segs": 原文对应的句子分隔,
					"tgt": 标题对应的wordPiece id
				},
				{
					"src_txt": 原文段落2,
					"tgt_txt": "",
					src_tokens: 原文对应的wordPiece id,
					src_cls: 原文对应的句子分隔,
					tgt_tokens: 标题对应的wordPiece id
				}
			]
		},
		{
			"heading": [{
				"src_txt": 原文段落1,
				"tgt_txt": 标题1.2,
				"src": 原文对应的wordPiece id,
				"segs": 原文对应的句子分隔,
				"tgt": 标题对应的wordPiece id
			}]
		}
	]
}
```


## Model 方案
读取一个segment中的所有Heading,按照上面的格式假设有3个标题，针对这三个标题可以做的处理
+ 判断是否含有多个段落，若有，送入bert后，得到张量，对齐Para1和para2,并得到一个para1+para2的加和张量信息
+ 重复上面的操作，得到每一个标题下面的所有张量加和的信息
+ 再分别对每一个张量加和的信息，又可分为以下几步:
    - 除第一个以外的所有张量，进行title的生成，并计算loss
    - 计算第一个时，需要把其他张量进行加和并求取平均值后，和现在的张量加到一起，进行title的生成，并计算loss
