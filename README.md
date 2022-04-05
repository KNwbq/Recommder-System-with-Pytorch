
## Recommender System with Pytorch

开源项目`Recommender System with Pytorch`主要是对经典的推荐算法论文进行复现，既包括CTR模型，也包括序列推荐模型。本项目启发于 [ZiyaoGeng/Recommender-System-with-TF2.0](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0)

**建立原因：**

1. 理论和实践似乎有很大的间隔，学术界与工业界的差距更是如此；
2. 更好的理解论文的核心内容，增强自己的工程能力；

**项目特点：**

- 使用Pytorch进行复现；
- 每个模型都是相互独立的，不存在依赖关系；
- 模型基本按照论文进行构建，实验尽量使用论文给出的的公共数据集；
- 模型都附有`README.md`，对于模型的训练使用有详细的介绍；
- 模型对应的论文有相应的论文报告；
- 代码源文件参数、函数命名规范，并且带有标准的注释；

**实验：**

| Metric    | Caser(50epoch) | Caser_v2(50epoch) | Caser_v2(100epoch) | Caser_v3(50epoch) | Caser_v3(100epoch) |
| --------- | -------------- | ----------------- | ------------------ | ----------------- | ------------------ |
| Prec@1    | 0.2929         | 0.3088            | 0.3157             | 0.3180            | 0.3177             |
| Prec@5    | 0.2539         | 0.2695            | 0.2691             | 0.2670            | 0.2695             |
| Prec@10   | 0.2271         | 0.2394            | 0.2393             | 0.2379            | 0.2386             |
| Recall@1  | 0.0184         | 0.0189            | 0.0191             | 0.0199            | 0.0198             |
| Recall@5  | 0.0780         | 0.0806            | 0.0799             | 0.0805            | 0.0813             |
| Recall@10 | 0.1343         | 0.1390            | 0.1395             | 0.1378            | 0.1392             |
| MAP       | 0.1748         | 0.1798            | 0.1802             | 0.1795            | 0.1810             |
