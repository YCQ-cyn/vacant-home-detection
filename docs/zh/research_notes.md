# 研究参考

**中文** | [English](../en/research_notes.md)

以下记录方法来源、所解决的业务问题及采用依据。文献本身不构成本项目的识别效果证据。

| 来源 | 解决的问题与选择 |
| --- | --- |
| [Bagging PU](https://doi.org/10.1016/j.patrec.2013.06.007) | U混有未标空置，采用抽样U的bagging对照，U不变成正常真值 |
| [SAR-PU](https://arxiv.org/abs/1808.08755) | 历史选择与明显静默、功率、间歇性相关；倾向加权仅作消融，P+U机制不完全可识别 |
| [nnPU](https://arxiv.org/abs/1703.00593) | 防止PU负风险过拟合；小MLP仅做敏感性，prior不确定 |
| [Elkan–Noto](https://doi.org/10.1145/1401890.1401920) | P/U学习基础；不把未知标注机制下的输出当真概率 |
| [Class prior](https://doi.org/10.1007/s10994-016-5589-8) | prior仅用于敏感性，不称真实空置率 |
| [TCN](https://arxiv.org/abs/1803.01271) | 小型膨胀卷积学习多尺度行为；只比较32/64宽度 |
| [TS2Vec](https://doi.org/10.1609/aaai.v36i8.20881)、[作者代码](https://github.com/zhihanyue/ts2vec) | 多尺度上下文对比；保留时钟对齐，不用随意平移、扭曲、shuffle |
| [主动学习综述](https://minds.wisconsin.edu/handle/1793/60660) | 有限人工预算下保留随机盲选与多样性定向挑战，不只挑最高分 |
| [UK-DALE](https://doi.org/10.1038/sdata.2015.7) | 参考负荷和NILM建模，测量语义和生活方式不同，不混入主训练 |
| [REDD](http://redd.csail.mit.edu/)、[REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)、[Pecan Street](https://www.pecanstreet.org/dataport/) | 研究数据迁移可行性；第一版优先内部自监督，不依赖未验证域迁移 |

原业务Word报告仅提供常驻负荷、低功率真实居住、自动背景、稀疏返访和周末使用等现象启发。不复刻其旧XGBoost与固定电量判定。低频冻结样本可能漏采、混叠，不能普遍称15分钟平均稀释。
