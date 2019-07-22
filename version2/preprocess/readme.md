# 方案二——数据预处理

## src和tgt的数据格式
原始数据处理成的src和tgt的格式为：
```
src.txt                            tgt.txt
1. title content                   para1
xxxxxx                             para2
1.1 title content                  para1
xxxxxx                             para2
xxxxxx                             para3

2. titile content                  para1
2.1 titile content                 para2

3. titile content                  para1
......
```

Note:实际的src的数据是不会带标号的，这里只是为了形象说明


## 数据预处理
数据处理成的最终格式如下，下面为一个segment的形式：
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
					"tgt_txt": 标题2,
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
				"tgt_txt": 标题3,
				"src": 原文对应的wordPiece id,
				"segs": 原文对应的句子分隔,
				"tgt": 标题对应的wordPiece id
			}]
		}
	]
}
```
格式说明：
segment：一个完整的章节，包括其子章节的内容.一个segment中包含多个heading
heading：一个完整的标题，包括其标题下所有的段落


