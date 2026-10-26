# 方法参考

**中文** | [English](../en/references.md)

以下列出方法来源及其用途。文献本身不构成本项目的识别效果证据。

| 来源 | 解决的问题与选择 |
| --- | --- |
| [Bagging PU](https://doi.org/10.1016/j.patrec.2013.06.007) | U混有未标空置，采用抽样U的bagging对照，U不变成正常真值 |
| [SAR-PU](https://arxiv.org/abs/1808.08755) | 解释历史正样本的选择偏差及倾向加权；仅凭P+U不能完全识别标注机制 |
| [nnPU](https://arxiv.org/abs/1703.00593) | 非负PU风险估计；类别先验不确定时需检查敏感性 |
| [Elkan–Noto](https://doi.org/10.1145/1401890.1401920) | P/U学习基础；不把未知标注机制下的输出当真概率 |
| [Class prior](https://doi.org/10.1007/s10994-016-5589-8) | prior仅用于敏感性，不称真实空置率 |
| [TCN](https://arxiv.org/abs/1803.01271) | 小型残差膨胀卷积学习多尺度行为，膨胀率为1、2、4、8 |
| [TS2Vec](https://doi.org/10.1609/aaai.v36i8.20881)、[作者代码](https://github.com/zhihanyue/ts2vec) | 多尺度上下文对比；保留时钟对齐，不用随意平移、扭曲、shuffle |
| [主动学习综述](https://minds.wisconsin.edu/handle/1793/60660) | 有限人工预算下保留随机盲选与多样性定向挑战，不只挑最高分 |
| [UK-DALE](https://doi.org/10.1038/sdata.2015.7) | 参考负荷和NILM建模，测量语义和生活方式不同，不混入主训练 |
| [REDD](http://redd.csail.mit.edu/)、[REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)、[Pecan Street](https://www.pecanstreet.org/dataport/) | 参考负荷建模；采用内部自监督，不依赖未验证域迁移 |

低频冻结样本可能漏采、混叠；未经设备语义确认，不假定完整区间平均，也不把功率积分代理称为真实电量。
