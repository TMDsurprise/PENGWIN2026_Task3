# [Algorithm Description] Task 3 Team OBL Lab

> 本文描述最后一次冻结提交版本：`e27 exact seed42 epoch 11 + fixed outer C2`。
> 文中 epoch 编号按训练框架从 0 开始，因此 `epoch=24` 表示完成了 25 轮训练。

## 1. Task

Task 3: PENGWIN-Reduction（骨盆骨折碎片复位）。

## 2. Team name

OBL Lab

## 3. Authors

Zhengliang Li, Tianyun Gu, Nan Zheng, Chunjie Xia, Wanxin Yu, and Yangyang Yang

## 4. Affiliations

1. School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China
   上海交通大学生物医学工程学院，中国上海
2. School of Intelligent Sports Engineering, Shanghai University of Sport, Shanghai, China
   上海体育大学智能体育工程学院，中国上海

Zhengliang Li、Tianyun Gu、Nan Zheng、Chunjie Xia 和 Wanxin Yu 隶属于单位 1；
Yangyang Yang 隶属于单位 2。

## 5. Contact author and email address

Zhengliang Li, lizhengliang@sjtu.edu.cn

## 6. Algorithm name or title

**OBL Reduction V4**

方法副标题：**Small-Fragment-Aware Multi-Candidate Assembly Transformer with Conservative Risk Gating**

中文副标题：**面向小碎片的多候选骨盆复位 Transformer 与保守风险门控**

Grand Challenge：<https://grand-challenge.org/algorithms/obl-reduction-v1/>

### 缩写与符号说明

- **PENGWIN**：Peripelvic Fracture Segmentation and Reduction Planning Challenge，骨盆周围骨折分割与复位规划挑战。
- **OBL Lab**：本参赛团队名称；本文不对 OBL 另作扩展。
- **SA**：Sacrum，骶骨。
- **LI**：Left Innominate bone，左侧髋骨（左半骨盆）。
- **RI**：Right Innominate bone，右侧髋骨（右半骨盆）。
- **ID**：Identifier，标识符；fragment ID 表示碎片编号。
- **OBJ**：Wavefront OBJ，输入三角网格文件格式。
- **JSON**：JavaScript Object Notation，最终刚体矩阵的结构化输出格式。
- **MSE**：Mean Squared Error，均方误差。
- **Smooth-L1**：平滑 L1 损失，也称 Huber 型回归损失。
- **CE / BCE**：Cross-Entropy / Binary Cross-Entropy，交叉熵/二元交叉熵损失。
- **SVD**：Singular Value Decomposition，奇异值分解；用于 Kabsch/Horn 刚体解算。
- **SE(3)**：Special Euclidean Group in 3D，三维特殊欧氏群，表示三维旋转和平移组成的刚体变换。
- **6D rotation**：连续六维旋转表示；它是旋转参数化方式，不表示物体具有六个旋转自由度。
- **DDP**：Distributed Data Parallel，PyTorch 多 GPU 数据并行训练。
- **GPU / CPU**：Graphics Processing Unit / Central Processing Unit，图形处理器/中央处理器。
- **LR**：Learning Rate，学习率。
- **RA**：Rotation-Aware，本项目的旋转感知训练阶段。
- **OHM**：Online Hard-example Mining，在线困难样本挖掘。
- **OOF**：Out-of-Fold，折外预测；每个患者仅由训练时未见过该患者的模型产生校准预测。
- **SFQ**：Small-Fragment Query，小碎片查询分支。
- **SFQ-F**：SFQ 的 Fragment self-attention 版本，即等权碎片 token 自注意力分支。
- **SFQ-FX**：在 SFQ-F 上增加 Cross-attention 的版本，使碎片查询完整骨盆 patch memory。
- **B1 / B2 / B3**：本项目 e26 阶段的三个互补候选分支编号，并非通用网络名称。
- **E2、e3、e25、e26、e27**：本项目实验和 checkpoint 的内部版本标识，不代表 epoch 数或领域标准模型。
- **C2**：本项目对第二阶段保守候选比较门控（stage-2 conservative candidate-comparison gate）的内部简称；它根据预测收益、严重失败风险和回退条件决定是否采用非安全候选。
- **TRE**：Target Registration Error，目标配准误差，单位为 mm。
- **Trans**：Translation Error，平移误差，单位为 mm。
- **Rot**：Rotation Error，旋转误差，单位为 degree（度）。
- **CD**：Chamfer Distance，Chamfer 距离；本文官方尺度结果以 mm 报告。
- **PA**：Part Accuracy，碎片准确率；表示满足赛方表面距离阈值的 fragment 比例。
- **Qsmall**：本项目按尺寸划分的小碎片子集，不是通用医学分型。
- **Area64**：本项目的表面积采样策略；按三角面面积分配点数，并保证每个 fragment 至少 64 点。
- **alpha（α）**：迭代位姿更新的步长缩放系数。
- **4 x 4 matrix**：齐次刚体变换矩阵，包含 3 x 3 旋转、3 x 1 平移和齐次坐标行。

## 7. Method description

我们将 Task 3 建模为基于配对坐标回归的多碎片刚体装配问题。输入是一个患者的骶骨（SA）、
左髋骨（LI）和右髋骨（RI）骨折碎片表面网格，输出是每个 fragment 的 `4 x 4` 刚体复位矩阵。
整个算法由两个部分组成：第一部分是逐点预测复位坐标的坐标主干；第二部分是主干后的候选生成、
候选排序和保守风险控制。

### 7.1 第一部分：坐标主干

坐标主干基于 PENGWIN 2026 Task 3 Baseline 的 Assembly Transformer。网络联合处理病例中的全部
碎片点、表面法向、fragment ID 和 bone type，通过全局点级 attention 为每个输入点预测其复位后
的配对目标坐标。对于每个 fragment，使用源点与预测目标点之间的 SVD Kabsch/Horn 解算旋转和平移。
该过程最多迭代 10 次；坐标更新量低于 2 mm 时提前终止。

最终 e25 主干具有以下连续训练血缘：

1. **Clean Baseline：**在 challenge simulation 上从零训练至 `epoch=659`，即 660 轮。
2. **Rotation-aware continuation：**从 Clean e659 初始化，最终采用 `epoch=19`，即继续训练 20 轮。
   除逐点坐标 MSE 外，引入 fragment-balanced coordinate、centered vector、长基线 point-pair、
   cross-covariance 以及可微 Horn/Kabsch Rot/Trans 监督，使旋转梯度直接进入坐标预测头。
3. **Rotation anti-forgetting OHM：**从 rotation-aware e19 初始化，最终采用 `epoch=24`，即继续训练
   25 轮。训练按 bone type、fragment size 和历史 failure mode 维护困难样本池，并加入 replay 与
   rotation-teacher consistency，降低只学习困难样本造成的灾难性遗忘。

因此，最终 e25 坐标主干顺序继承了 `660 + 20 + 25 = 705` 个训练 epoch。三个阶段均在双 RTX 5090、
DDP 和 mixed precision 环境中完成。不同阶段的数据采样与损失不同，因此 705 轮表示真实权重血缘，
而不是 705 轮完全相同的优化。

推理输入采用 Area64 采样：SA、LI、RI 每个 bone 分别采样 5000 个表面点，骨内按三角面面积分配，
并保证每个 fragment 至少 64 点。Area64 是确定性预处理策略，不产生额外训练 epoch。

### 7.2 第二部分：候选生成、排序和安全决策

#### 7.2.1 原始四候选与 e3 Ranker

原始 e3 Ranker 的四个候选槽位在训练实现中定义为 `alpha=[0, 0.5, 1.0, 1.25]`。冻结提交的
exact-full 在线链根据 Clinical170 OOF 校准将实际 source alpha 固定为 `[0, 1.0, 1.25, 1.25]`；
后续外层 e27 对四个异构完整骨盆候选使用 `[0, 1.0, 1.0, 1.0]` 作为 candidate-source 编码。
原始 Ranker 并非从零训练，其真实血缘为：历史 E2 Ranker `epoch=119`（120 轮）作为初始化；随后冻结 e25
coordinate backbone，在双 RTX 5090 上对 Ranker 进行 30 轮 e25-specific strict recalibration，
得到 `epoch=29`；再以更低学习率继续训练，最终选择 continuation 的 `epoch=3`，即额外 4 轮。
因此最终原始 e3 Ranker 的参数血缘累计经历 154 轮优化，其中最后 34 轮专门适配 e25 主干。

导入历史 e119 时只加载 Ranker 参数，并明确排除旧 checkpoint 中的 coordinate-backbone 参数，
避免历史 OHM 主干覆盖 e25。原始 Ranker 与内部 C2 共同产生一个稳定的 baseline-selected safety pose。

#### 7.2.2 SFQ-F、SFQ-FX 与 exact-full

SFQ-F 为每个 fragment 构造等权 token，并使用 fragment-level self-attention，使小碎片不会因点数少
而在全局点级表示中被大碎片淹没。SFQ-FX 在此基础上增加 fragment query 到完整骨盆 patch memory
的 cross-attention。两条分支均通过零初始化的 gated `SE(3)` residual head 对 e25 位姿进行保守修正。

SFQ-F 和 SFQ-FX 每条分支均先进行 3 轮 screen/preheat，再以 seed 42、低学习率继续训练 9 轮，
因此最终每条分支经历 12 轮正式优化。它们在四卡 RTX 5090 服务器上以“一项实验占用一张 GPU”方式
并行训练，不是每条分支使用双卡 DDP。32-case、100-epoch capacity overfit 只用于验证模型是否能学动，
没有进入提交权重。

原始四候选与 SFQ 并不是二选一。原始 e3 Ranker+C2 的输出始终保留为安全回退；SFQ-F/FX 只提供
附加的 fragment-aware 修正候选。低容量 Huber-ridge 校准器在 patient-level five-fold OOF 后进行
full fit，并与 `margin=0.15` 的内部 C2 决定是否接受 SFQ 修正，形成 exact-full 安全候选。Huber-ridge
校准和 C2 不属于神经网络 epoch 训练。

#### 7.2.3 e26 互补候选

除 exact-full 外，算法保留三个结构或训练分布互补的完整骨盆候选：

- **B1：**在 e25 后部加入 equal-fragment context adapter；训练完整运行 3 轮，选择 `epoch=1`。
- **B2：**在 B1 基础上加入 point-reliability auxiliary 与稳健 fragment-coordinate loss；训练完整
  运行 3 轮，选择 `epoch=2`。最终刚体解算仍使用统一权重 Kabsch/Horn。
- **B3：**在 B2 基础上加入 Qsmall hard sampling、replay 和 OHM；训练完整运行 8 轮，选择
  `epoch=6`。

B1、B2、B3 在四卡 RTX 5090 服务器上分别占用一张 GPU 并行训练，而不是各自使用双卡。

#### 7.2.4 e27 完整骨盆组合 Ranker 与外层 C2

最终外层候选池为：candidate 0 = exact-full，candidate 1 = B1，candidate 2 = B2，candidate 3 = B3。
四个候选分别完成整套骨盆迭代，并被映射到同一 canonical state。e27 Ranker 联合评价四套完整骨盆
组合，而不是孤立评价一个点或一个 fragment。它使用主干 fragment feature、candidate geometry、
bone/candidate/fragment embedding 和三层 context Transformer，输出候选分数、TRE/Trans/Rot/CD
质量 proxy 及 severe-failure 概率。

e27 在 8617 个 simulation training samples 和 958 个 patient-held-out validation samples 上训练，
每个样本缓存四路最多 10 次 rollout。配置最大 30 轮、每轮 validation、early-stop patience 8；最终
冻结 `LR=2e-5, seed=42, epoch=11`，即采用第 12 轮权重。最大 30 轮只是训练上限，不能写成实际
采用了 30 轮。

外层 C2 只有在新候选的预测综合误差至少改善 0.05、severe risk 不增加且风险概率不超过 0.2 时
允许替换 candidate 0；patient-level rollback 在整例风险上升时回退。C2 是固定决策规则，无训练 epoch。

## 8. Main technical contributions and/or novel components

1. **Area64 小碎片保护采样：**按表面积分配点数并保证每个 fragment 至少 64 点。
2. **配对坐标驱动的旋转感知训练：**将 centered vector、长基线 pair、cross-covariance 与可微
   Horn/Kabsch 监督直接作用于 coordinate head。
3. **Anti-forgetting OHM：**按 bone、size 和 failure mode 分池采样困难样本，并通过 replay 和
   teacher consistency 保护普通病例性能。
4. **Small-Fragment Query：**通过等权 fragment token、完整骨盆 cross-attention 和零初始化
   `SE(3)` residual，为小碎片提供独立的全局查询与修正能力。
5. **异构完整骨盆候选池：**同时保留 exact-full、fragment-context、point-reliability 和 hard-replay
   四种完整组合，而不是只增加相似 checkpoint。
6. **完整骨盆组合 Ranker：**联合评价 SA/LI/RI 的整体几何一致性、指标 proxy 和严重失败风险。
7. **双层保守风险门控：**exact-full 内部门控与最终 outer C2/patient rollback 均保留显式安全回退。

## 9. Complete pipeline

1. 解析输入 OBJ 中的 SA、LI、RI、fragment ID、表面点和法向。
2. 每骨采样 5000 点，按表面积分配，每个 fragment 至少 64 点。
3. 对病例中心化并做各向同性尺度归一化。
4. 使用 e25 坐标主干预测逐点配对目标坐标，并用 Kabsch/Horn 求解 fragment 刚体更新。
5. 使用冻结部署的 source alpha `[0,1.0,1.25,1.25]` 运行四候选迭代，由 e3 Ranker+C2 得到安全基准位姿。
6. 使用 SFQ-F/FX 生成小碎片修正候选，通过内部校准和 C2 得到 exact-full candidate 0。
7. 独立运行 e26 B1、B2、B3，得到 candidates 1-3。
8. 将四套完整骨盆位姿映射到同一 canonical state。
9. 使用 e27 Ranker 预测候选排序、指标 proxy 和 severe-failure risk。
10. 使用 outer C2 和 patient-level rollback 决定是否替换 candidate 0。
11. 以 SA fragment 1 为锚归一化累计变换。
12. 输出每个 fragment 的 `4 x 4` reduction pose matrix JSON。

## 10. Use of external data

未使用外部数据集。神经候选生成器和 e27 Ranker均使用 PENGWIN 2026 提供的 simulation 数据训练。
Challenge 提供的 Clinical170 training cases 用于方法消融、patient-level five-fold OOF、低容量
Huber-ridge/C2 校准及冻结链路回放，因此不属于外部数据。Clinical 标签没有用于梯度更新 e25 坐标
主干、SFQ 候选生成器或 e27 神经 Ranker。隐藏测试集及标签始终不可见。

## 11. Use of externally pretrained models

未使用 PointMAE、TotalSegmentator 或其他外部预训练模型和外部权重。所有初始化 checkpoint 均来自
本团队使用 challenge-provided simulation 训练的模型。

## 12. Preprocessing techniques

- 解析 OBJ 的 bone type 与 fragment ID，并从三角网格均匀采样表面点和法向。
- SA、LI、RI 每骨固定 5000 点；骨内按三角面面积分配，每个 fragment 最少 64 点。
- 完整病例中心化和各向同性尺度归一化。
- 每例固定 NumPy/PyTorch seed 42，保证采样和推理可复现。
- 构造 fragment count、直径、质心、协方差、bone type 等元数据。
- 输出前以 SA fragment 1 为刚体锚归一化。

## 13. Data augmentation techniques

- challenge simulation 在线生成的随机碎片组合与随机 `SE(3)` 旋转和平移。
- 随机 anchor、fragment merge、fragment count 与 fracture pattern。
- 点丢弃、region dropout、表面重采样和坐标 jitter。
- 按 SA/LI/RI、fragment size 和历史 failure mode 分池采样。
- Qsmall hard sampling 与 Clean/e25 replay。
- 不同尺寸和不同刚体偏移范围的条件化采样。

## 14. Training and validation strategy

### 14.1 最终权重真实训练链路

| 部件 | 初始化 | 运行/采用 epoch | 硬件 | 是否进入提交 |
|---|---|---:|---|---|
| Clean Baseline | 从零 | 运行并采用 660 轮，`e659` | 双 RTX 5090 DDP | 是 |
| Rotation-aware | Clean e659 | 采用 20 轮，`e19` | 双 RTX 5090 DDP | 是 |
| Anti-forgetting OHM | RA e19 | 采用 25 轮，`e24` | 双 RTX 5090 DDP | 是 |
| 历史 E2 Ranker | 历史 simulation 训练 | 120 轮，`e119` | 双 RTX 5090 | 作为初始化 |
| e25 Ranker strict recalibration | E2 e119 Ranker heads | 30 轮，`e29` | 双 RTX 5090 DDP | 是 |
| e25 Ranker low-LR continuation | e29 | 采用 4 轮，`e3` | 双 RTX 5090 DDP | 是 |
| Area64 | 无 | 0 轮 | CPU/GPU 预处理 | 是 |
| SFQ-F | e25 | 3 轮 screen + 9 轮 continuation = 12 轮 | 单张5090，并行运行 | 是 |
| SFQ-FX | e25 | 3 + 9 = 12 轮 | 单张5090，并行运行 | 是 |
| exact-full Huber-ridge/C2 | OOF/full fit | 0 个神经 epoch | 低容量拟合 | 是 |
| e26 B1 | e25 | 运行3轮，选`e1` | 单张5090 | 是 |
| e26 B2 | B1/e25 | 运行3轮，选`e2` | 单张5090 | 是 |
| e26 B3 | B2/e25 | 运行8轮，选`e6` | 单张5090 | 是 |
| e27 full Ranker | 冻结四候选池 | 最大30轮，采用`e11`即第12轮 | 单张5090训练，四实验并行 | 是 |
| outer C2 | 固定规则 | 0轮 | CPU | 是 |

主干累计训练血缘为 705 轮。原始 e3 Ranker 的参数血缘为 120 + 30 + 4 = 154 轮。SFQ、e26、
e27 是主干后的独立候选生成或选择模块，不能加到主干 epoch 上宣称一次端到端训练。

### 14.2 验证与模型选择

- Clean/RA/OHM 使用 patient-held-out simulation validation 与 early stopping/milestone selection。
- e25 Ranker 冻结坐标主干，以 simulation `val/oracle_regret` 选择 checkpoint。
- SFQ 使用固定 simulation validation、零控制和 Clinical170 candidate-oracle 诊断；capacity overfit
  只验证可学习性，不作为泛化证据。
- Clinical170 采用 patient-level five-fold OOF；sealed patients 不参与梯度、early stop 或门控搜索。
- e27 使用 simulation 8617/958 patient split，每轮 validation，patience 8，冻结 seed42 e11。
- 隐藏测试标签没有参与任何训练、选择或推理。

冻结提交包在 170 个 challenge-provided clinical training cases、1014 个 fragments 上的 official-scale
回放结果为：

| TRE (mm) | Trans (mm) | Rot (deg) | PA | CD (mm) |
|---:|---:|---:|---:|---:|
| 3.151657 | 3.857754 | 5.167839 | 0.812782 | 3.681537 |

Clinical170 参与了低容量校准和候选消融，因此该结果是冻结部署链的训练域回放，不等同于隐藏测试性能。

## 15. Loss function(s)

### 15.1 坐标主干

基础坐标项不是在完整 point-wise MSE 上额外叠加 `0.25` 倍 fragment loss，而是二者的凸组合：

```text
L_coord = 0.75 * L_global_point_MSE
        + 0.25 * L_fragment_balanced_MSE

L_RA = L_coord + s(epoch) * (
          0.010  * L_centered_vector
        + 0.004  * L_long_baseline_pair
        + 0.005  * L_cross_covariance
        + 0.010  * L_Horn_rotation
        + 0.0015 * L_Horn_translation
       )
```

其中 `s(epoch)` 是 rotation auxiliary warm-up。OHM 阶段将 `L_coord` 与按病例难度加权的
`L_hard` 混合，混合权重 warm-up 后最大为 `0.30`；上述 rotation auxiliary 始终保留，并且每
4 个 step 以权重 `0.020` 施加一次仅针对可靠自然样本的 selective rotation-teacher consistency。

### 15.2 SFQ residual

```text
L_SFQ = 2.0  * L_rotation_chordal
      + 0.5  * L_translation_SmoothL1
      + 0.5  * L_paired_coordinate_SmoothL1
      + 0.2  * L_preserve
      + 0.02 * L_residual_regularization
```

前三项采用 `w_i = 1 + 2 * qsmall_strength_i` 的 fragment 加权平均；translation 和 paired-coordinate
误差先换算为毫米并除以 30 mm 归一化。preserve loss 惩罚相对 e25 base pose 的 Rot/Trans/TRE
恶化。通用 SFQ 代码还支持 correction-confidence BCE 和 point-confidence KL，但冻结提交只使用
F/FX 变体，未启用 G/H confidence heads，因此这两项在上传权重中严格为零。

### 15.3 Candidate Ranker

候选 utility 使用 TRE、Trans、Rot、CD 按榜单参考尺度归一化。Ranker 损失为：

```text
L_rank = 1.0  * L_hard_oracle_CE
       + 0.5  * L_pairwise_ranking
       + 0.25 * L_metric_regression
       + 0.2  * L_severe_BCE
```

其中 hard-oracle 是归一化 utility 最低的候选；pairwise loss 只监督 utility 差异大于 `0.05` 的
候选对。metric head 使用 Smooth-L1 回归 `log(1 + metric)`，推理时通过 `expm1` 恢复；CD 是每个
候选采样 64 点计算的几何 proxy。最终 utility 不包含 PA，severe 标签定义为 `Rot > 30 deg` 或
`Trans > 20 mm`。
Huber-ridge calibration 和 C2 是低容量/确定性后处理，不产生上述神经网络损失。

## 16. Base network architecture

- Assembly Transformer coordinate backbone：384维、12层、8个 attention heads。
- 输出：逐点复位配对坐标；刚体解算：per-fragment SVD Kabsch/Horn。
- SFQ-F/FX：384维 fragment token、两层8-head fragment Transformer、可选 pelvis patch
  cross-attention、连续6D rotation + 3D translation residual head。
- e26 B1：equal-fragment context adapter。
- e26 B2：B1 + point reliability auxiliary head。
- e26 B3：B2结构配合 Qsmall hard/replay/OHM 训练。
- e27 Ranker：27维 candidate geometry descriptor、bone/candidate/fragment embeddings、三层
  hidden-dim 256、8-head context Transformer，以及 score/metric/severe heads。

## 17. Ensembling strategies used during inference

未使用 checkpoint 权重平均或矩阵平均。最终属于候选选择式集成：exact-full、B1、B2、B3 分别生成
四套完整位姿，e27 Ranker 联合比较；candidate 0 始终为显式 fallback。只有通过 metric margin、
severe-risk threshold 和 patient rollback 的非零候选才会被接受。exact-full 内部 C2 使用
`margin=0.15`；outer C2 使用 `margin=0.05, severe tolerance=0, severe threshold=0.2`。

## 18. Public code repository

代码仓库：<https://github.com/TMDsurprise/PENGWIN2026_Task3>

## 19. Relevant references

1. Sutuk. *PENGWIN2026 Task 3 Reduction Baseline*.
   https://github.com/Sutuk/PENGWIN2026_Task3_Reduction_Baseline
2. Vaswani, A., et al. *Attention Is All You Need*. NeurIPS, 2017.
3. Kabsch, W. *A solution for the best rotation to relate two sets of vectors*. Acta
   Crystallographica Section A, 1976.
4. Horn, B. K. P. *Closed-form solution of absolute orientation using unit quaternions*.
   Journal of the Optical Society of America A, 1987.
5. Zhou, Y., et al. *On the Continuity of Rotation Representations in Neural Networks*. CVPR, 2019.
6. Shrivastava, A., Gupta, A., and Girshick, R. *Training Region-Based Object Detectors with Online
   Hard Example Mining*. CVPR, 2016.

---

## 发布前检查清单（不要贴入论坛）

- [ ] 补全第5项联系人姓名和邮箱。
- [ ] 由所有作者确认作者顺序和单位写法。
- [x] 公开 Clinical170 内部回放指标。
- [x] 第18节已填写公开代码仓库地址。
- [ ] 发布前删除本检查清单，并保留论坛帖子链接或截图。
