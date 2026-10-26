# 本地数据接口

**中文** | [English](../en/data_contract.md)

工作区位于源码目录之外；下列路径相对`VACANT_HOME_WORKSPACE/recognition`。仓库不附带真实文件。pickle/joblib仅载入可信本地产物。

## 必需的预处理输入

| 路径 | 内容 |
| --- | --- |
| `inputs/power_samples.xlsx` | 原始表，Sheet1含表号、供电单位、硬件元数据与时钟列 |
| `inputs/legacy_labels.xlsx` | 第一列表号、第二列历史空置类型 |
| `inputs/training_scope.csv` | 拟合资格、历史证据来源、保护状态 |
| `inputs/data_snapshot/daily_curves.pkl` | 清洗日曲线、日期、96点与质量字段 |
| `inputs/semantics_robust_core.csv` | 已准备的语义稳健逐户特征 |
| `inputs/automatic_human_axes.csv` | 保护池设计前的scout轴 |
| `inputs/scout_screening.csv` | 冻结scout结果，用于保护池设计 |
| `inputs/daily_inputs.npz` | float32的x，日行数×8通道×96点 |
| `inputs/daily_input_index.csv` | 与x严格对齐的表号和日期 |

训练需要完整的预处理输入，包括特征、来源信息、标签和初步筛查结果。从原始数据生成这些输入的预处理流程位于本包接口之外。仅源码可运行测试和检查方法；真实训练需提供完整私有输入。

## 日曲线

列含`meter_id`、`date`、`utility`、`0:00`到`23:45`。质量列含`unresolved_missing_points`和`original_missing_points`。源文件可能的24:00不是第97个编码点。

8通道依次为：带符号log功率、稳健归一化形态、带符号log样本差、相对样本差、原缺测mask、插值mask、未解决缺测mask、负值mask。编码张量保持有限值，同时保留质量mask，不能把缺测静默当全零。

## 拟合范围

`training_scope.csv`含`meter_id`、`data_invalid`、`strict_all_zero`、`model_a_fit`、`shadow_fit_excluding_gold_test`、`literal_transductive_full_fit`、`legacy_high_hard`、`legacy_medium_soft`、`legacy_low_diagnostic_only`、`protected_cohort`。

prepare生成`fit_scope.csv`、`future_human_holdout_manifest.csv`和冻结hash。保护池30户（10随机、20定向）必须在SSL和任何参考拟合前冻结；已有保护池不能替换。

## 推理

特征表包含表号及拟合后的鲁棒/embedding列。本地模型为`models/final_pu_channels.joblib`和`models/conditional_subtype.joblib`，保存预处理与固定参考分布。

`infer`输出PU证据和四类条件支持；完整路由、稳定性、冲突和业务类别由validation/report流程产生。专项需本地最终embedding、逐户结果、fit scope、保护池和历史正样本，输出到独立high_power工作区。
