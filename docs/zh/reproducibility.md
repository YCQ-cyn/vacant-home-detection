# 复现说明

**中文** | [English](../en/reproducibility.md)

## 无数据验证

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m vacant_home --help
```

测试采用程序生成的数组和虚拟表号，验证模型接口、隔离、门槛、序列缩放与序列化；不测真实空置准确率，不依赖相邻历史目录。

## 私有训练

提供[数据接口](data_contract.md)中的预处理输入，并使用新工作区，保留现有冻结产物。

```powershell
$env:VACANT_HOME_WORKSPACE = 'D:\Private\vacant-home-training'
python -m vacant_home train --stage prepare
python -m vacant_home train --stage ssl
python -m vacant_home train --stage ablation
python -m vacant_home train --stage validation
python -m vacant_home train --stage report
python -m vacant_home high-power --fit --audit-amplitude
```

CPU训练可能较长，包括小规模表示消融和100次交叉验证置换。推理与同侪审计不会自动训练。固定种子、soft权重、prior和门槛在源码与专项决策规范中记录。日监督按户分组，保护用户不进入任何拟合或参考。

标签交叉验证不等于表示逐fold隔离；研究SSL为允许池内传导式评价。

## 冻结推理与审计

```powershell
python -m vacant_home --workspace 'D:\Private\vacant-home-trained' infer --features 'D:\Private\features.parquet' --output 'D:\Private\scores.csv'
python -m vacant_home peer-audit --targets YOUR_PRIVATE_TARGET --hardware 'D:\Private\hardware.csv' --scores 'D:\Private\scores.csv' --holdout 'D:\Private\holdout.csv' --output-dir 'D:\Private\peer-audit'
```

推理分位使用bundle内固定参考分布，不随批次改变。旧本地模型若缺参考，须输入完整拟合人群，否则报错。审计硬件列为表号、供电单位、电表厂家、版本、终端厂家；holdout输入全部保护ID。审计先冻结并hash匹配规范，不改分数与Tier。
