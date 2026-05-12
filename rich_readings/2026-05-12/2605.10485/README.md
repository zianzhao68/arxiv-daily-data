# VEGA: Visual Encoder Grounding Alignment for Spatially-Aware Vision-Language-Action Models

#

# 基本信息

- **arXiv:** 2605.10485
- **Authors:** Hao Wang, Xiaobao Wei, Jingyang He, Chengyu Bai, Chun-Kai Fan, Jiajun Cao, Jintao Chen, Ying Li, Shanyu Rong, Ming Lu, Xiaozhu Ju, Jian Tang, Shanghang Zhang
- **Categories:** cs.RO
- **一句话结论:** 通过在视觉编码器输出层与三维感知特征直接对齐，VEGA 在不增加推理开销的前提下，显著提升了 VLA 模型的空间感知能力与机器人操作性能，并在仿真与真实世界任务中达到隐式空间 grounding 方法的最新水平。

#

# 研究问题

当前 Vision-Language-Action (VLA) 模型的视觉骨干网络通常仅在二维图像-文本数据上进行预训练，缺乏显式的三维几何监督，导致其视觉表征在空间深度、视角变化与物体相对关系等维度上存在先天不足。这种“空间感知缺失”严重限制了机器人在需要精细几何推理的操作任务中的泛化性与可靠性。

与此同时，现有隐式空间 grounding 方法（如 Spatial Forcing、ROCKET）虽然尝试将三维基础模型的特征注入 VLA，但它们普遍依赖经验性的层搜索（layer search），并且对齐操作发生在 LLM 层级的视觉 token 上——此时空间结构已与语言语义高度纠缠，既降低了几何可解释性，也限制了对齐目标的清晰度。因此，本文核心要解决的是：**如何在避免语义纠缠与推理开销的前提下，将三维空间先验有效注入 VLA 的视觉表征阶段？**

#

# 任务与挑战

本文聚焦双臂与单臂机器人的精细操作任务，输入为多视角 RGB 图像与语言指令，输出为低层动作序列。实验覆盖 RoboTwin 2.0 仿真基准（六项双臂任务，含 Easy/Hard 两种设置）与真实世界 AgileX ALOHA 平台（四项单臂/双臂任务）。

主要挑战包括：

1. **二维预训练的局限：** 标准视觉编码器（如 DINOv2、SigLIP）缺乏三维几何先验，难以准确编码深度与空间布局。
2. **显式三维输入的代价：** 增加深度传感器或单目深度估计会引入硬件成本、标定负担、噪声传播与推理延迟。
3. **深层对齐的语义纠缠：** 在 LLM 中间层或深层视觉层进行特征对齐时，视觉特征已与语言上下文融合，空间结构不再纯粹。
4. **层搜索的不可泛化性：** 针对不同任务或模型架构，需暴力搜索最优对齐层，难以规模化迁移。

#

# 核心 Insight

VEGA 的核心思想极其简洁：**将空间对齐从“LLM 层级”前移到“视觉编码器输出层级”**。具体而言，作者认为视觉编码器是视觉信息进入语言模型之前的最后一道“纯视觉”关口；在此处与三维感知特征对齐，可以在语言嵌入发生之前就将空间意识注入表征，从而彻底避免几何结构与语言语义的纠缠。此外，由于对齐仅通过一个轻量投影器完成，该投影器在推理时可直接丢弃，实现零额外计算开销。

这一范式与现有方法形成鲜明对比：显式方法需要额外的深度估计网络；隐式 LLM 级方法需要在语言上下文中“找回”空间结构；而 VEGA 直接在视觉骨干的“出口”处建立几何监督，既保留了三维信息的纯净性，又维持了 VLA 推理流程的完整性。

![三种空间 grounding 范式的架构对比图](figures/figure-007-teaser.png)

上图清晰展示了三种范式的差异：(1) 显式空间 grounding 依赖深度估计器，增加推理负担；(2) 隐式 LLM 级 token 对齐在语言模型内部进行，空间与语义已纠缠；(3) VEGA 在视觉编码器输出层直接对齐，在语言嵌入前完成空间 grounding。

#

# 方法与公式

VEGA 基于 OpenVLA-OFT 构建，其视觉骨干包含 SigLIP 与 DINOv2 两个分支。VEGA 仅对负责细粒度空间表征的 DINOv2 分支进行监督。

**特征提取与对齐目标。** 给定输入图像 $I$，从 VLA 的 DINOv2 编码器倒数第二个 Transformer block 提取 patch 级特征 $\mathbf{F}^{\mathrm{DINO}}$（绕过 FiLM 语言条件，保留纯视觉表征）；同时，从冻结的 DINOv2-FiT3D 教师编码器最后一个 block 提取三维感知特征 $\mathbf{F}^{\mathrm{FiT3D}}$。二者维度均为 $\mathbb{R}^{N \times d}$（$N$ 为 token 数，$d=1024$）。

```math
\mathbf{F}^{\mathrm{DINO}} = \varepsilon^{\mathrm{DINO}}_{L-2}(I), \quad
\mathbf{F}^{\mathrm{FiT3D}} = \varepsilon^{\mathrm{FiT3D}}_{L-1}(I)
\tag{1}
```

**轻量投影器。** 通过一个由 LayerNorm 与两层 GELU-MLP 构成的投影器 $\phi$ 将学生特征映射到教师特征空间：

```math
\hat{\mathbf{F}}^{\mathrm{DINO}} = \phi(\mathbf{F}^{\mathrm{DINO}})
\tag{2}
```

投影器具体实现为：

```math
\phi(\mathbf{F}) = \mathbf{W}_2 \cdot \mathrm{GELU}(\mathbf{W}_1 \cdot \mathrm{LayerNorm}(\mathbf{F}) + \mathbf{b}_1) + \mathbf{b}_2
\tag{3}
```

其中 $\mathbf{W}_1, \mathbf{W}_2 \in \mathbb{R}^{1024 \times 1024}$，参数量仅约 2.1M，训练结束后即丢弃。

**对齐损失。** 采用平均余弦距离作为对齐目标：

```math
\mathcal{L}_{\mathrm{align}} = \frac{1}{N}\sum_{i=1}^{N} \left(1 - \frac{\hat{\mathbf{F}}^{\mathrm{DINO}}_i \cdot \mathbf{F}^{\mathrm{FiT3D}}_i}{\left\lVert \hat{\mathbf{F}}^{\mathrm{DINO}}_i \right\rVert \left\lVert \mathbf{F}^{\mathrm{FiT3D}}_i \right\rVert}\right)
\tag{4}
```

**总训练目标。** 与标准动作预测损失 $\mathcal{L}_{\mathrm{action}}$ 加权组合：

```math
\mathcal{L}_{\mathrm{VEGA}} = \mathcal{L}_{\mathrm{action}} + \lambda \mathcal{L}_{\mathrm{align}}
\tag{5}
```

实验中 $\lambda = 0.1$。动作预测损失遵循 OpenVLA-OFT 的原始定义：

```math
\mathcal{L}_{\mathrm{action}} = \mathcal{L}\left[\mathcal{G}(\{x^A_t\}_{t=1}^{T}),\ A_{\mathrm{gt}}\right]
\tag{6}
```

其中 $\mathcal{G}$ 为动作专家头（如两层 MLP 或 flow-matching 头），$A_{\mathrm{gt}}$ 为示教动作序列。

![VEGA 模型架构与训练流程图](figures/figure-006-method.png)

上图展示了完整数据流：多视角图像分别输入 SigLIP 与 DINOv2，DINOv2 分支经投影器与冻结的 FiT3D 教师对齐；对齐损失与动作预测损失联合训练；推理时仅保留标准 VLA 前向路径。

#

# 贡献拆解

1. **诊断现有隐式 grounding 的结构性缺陷并给出新对齐位置。**  
   作者系统指出当前隐式方法的两个 compounding issues：依赖经验性层搜索，以及在 LLM 层级对齐导致几何-语义纠缠。通过将监督信号直接施加在视觉编码器输出，VEGA 同时规避了层搜索需求与语义纠缠问题，提供了更具几何可解释性的对齐目标。

2. **提出零推理开销的轻量对齐框架。**  
   VEGA 仅引入一个约 2.1M 参数的两层 MLP 投影器，在训练阶段完成 DINOv2-FiT3D 空间知识的迁移；推理时完全丢弃投影器与教师网络。相比显式深度估计或跨层特征融合，该方法无需任何额外传感器、额外前向网络或复杂架构改动。

3. **在仿真与真实世界同时取得隐式 grounding 方法的 SOTA。**  
   在 RoboTwin 2.0 上，VEGA 于 Easy/Hard 设置分别取得 67.5% 与 30.7% 的平均成功率，较最强基线 OFT+SF 提升 3.3% 与 2.9%。在真实世界四项任务中平均成功率达 60%，优于 OFT+SF 的 55%。

4. **通过受控实验验证 FiT3D 的跨域迁移性与训练/数据效率。**  
   特征可视化（PCA/ARI）与预训练对比实验表明，FiT3D 的密集 patch 特征在机器人域仍保持清晰的对象边界与空间一致性；且 VEGA 在仅使用 25% 数据时即可带来约 10% 的绝对成功率提升，10k 训练步即可达到基线 60k 步性能。

#

# 关键图表解读

**图 1：三种空间 grounding 范式对比（figure-007-teaser.png）**  
该图是论文的“总览图”，直接支撑核心论点。左侧显式方法将深度图作为额外输入，增加推理链路与误差传播；中间隐式 LLM 级方法（如基于 VGGT）在 LLM 内部进行 token 对齐，此时视觉与文本 token 已交错，空间结构被语义上下文污染；右侧 VEGA 在视觉编码器输出端（进入 LLM 之前）完成对齐，空间特征保持独立且纯净。读图时应注意三种范式中“对齐箭头”的位置差异，这是理解 VEGA 设计哲学的关键。

**图 2：VEGA 框架与数据流（figure-006-method.png）**  
此图支撑方法细节。输入包含左腕、主视角、右腕三视角图像；SigLIP 与 DINOv2 分别提取特征；DINOv2 特征经 MLP Projector 与 FiT3D 教师对齐（$\mathcal{L}_{\mathrm{align}}$）；LLaMA 2 7B 融合视觉-语言 token 后预测动作，经 Action De-Tokenizer 输出。注意图中虚线表示仅在训练时存在的路径，说明推理零开销。

**图 3：训练效率与数据效率（figure-001-efficiency.png）**  
左图（Training Efficiency）显示 VEGA（绿色菱形）在 10k 步时成功率已接近 OpenVLA-OFT（红色圆形）60k 步的水平，且最终收敛到更高平台（约 0.78 vs 0.70）。右图（Data Efficiency）显示在 25%、50%、75%、100% 数据比例下，VEGA 均显著优于基线；在 25% 数据时差距最大（约 0.41 vs 0.31）。这组图支撑了“空间先验作为强归纳偏置可加速学习并降低数据需求”的论点。

![训练效率与数据效率对比曲线](figures/figure-001-efficiency.png)

**图 4：特征可视化与下游成功率（figure-003-probe.png）**  
左图（Feature Visualization）通过 PCA 展示 RoboTwin 2.0 与 Bridge Dataset V2 上的 patch 特征：DINOv2-FiT3D 相比原始 DINOv2 具有更清晰的对象边界与更一致的空间结构。右图（Downstream Task Success Rate）显示在 Move Playingcard Away 任务上，无论编码器是否冻结，DINOv2-FiT3D 均大幅优于原始 DINOv2；且冻结与可训练 FiT3D 差距极小，说明 FiT3D 特征本身已具备高度内在空间质量，无需任务特定微调即可迁移。

![不同 backbone 的下游任务成功率随训练步数变化曲线](figures/figure-003-probe.png)

#

# 实验与消融

**数据集与设定。** 仿真环境采用 RoboTwin 2.0（AgiLeX ALOHA 平台），覆盖 Move Playingcard Away、Turn Switch、Click Bell、Beat Block、Lift Pot、Place Shoes 六项双臂任务。Easy 使用官方干净示教数据，Hard 引入场景杂乱、背景纹理、光照变化与桌面高度变化等域随机化。真实世界在 AgileX ALOHA 上执行四项任务（Close Laptop、Handover Cucumber、Pick Dual Carrots into Dual Bowls、Pick Dual Flowers into Vase），每项 20 次试验。

**基线与指标。** 对比基线包括扩散策略 DP、RDT、$\pi_0$、$\pi_0$+SF、$\pi_0$+ROCKET、OpenVLA-OFT、OFT+SF。指标为任务成功率。

**主结果。** 在 RoboTwin 2.0 上（Tab. final_results），VEGA 在 Easy 设置平均成功率 67.5%，Hard 设置 30.7%，较最强基线 OFT+SF（64.2% / 27.8%）分别提升 3.3% 与 2.9%。在 Hard 设置下，VEGA 的优势更为显著，表明视觉编码器级的空间 grounding 对分布偏移具有更强鲁棒性。在真实世界（Tab. real_world_results），VEGA 平均成功率 60%，优于 OFT+SF（55%）与 OpenVLA-OFT（48%）。

**消融实验。**

- **对齐系数 $\lambda$：** $\lambda=0.1$ 时达到最佳平衡（Easy 67.5%，Hard 30.7%）。$\lambda=0.2$ 导致 Easy 下降至 61.3%，说明过强对齐会干扰动作学习；$\lambda=0.05$ 则 Hard 降至 21.7%，空间 grounding 不足。
- **空间教师选择：** 以 VGGT 作为教师时，Hard 设置成功率暴跌至 0.04，甚至低于无教师基线（0.34）。作者归因于 VGGT 的中间表征面向场景级新视角合成，与 ViT patch token 存在架构失配；而 FiT3D 基于 DINOv2 微调，提供同质的密集 patch 特征，对齐更稳定。
- **编码器预训练对比：** 在 Bridge Dataset v2 上预训练时，冻结 FiT3D 编码器在 Move Playingcard Away 上达到 0.62，接近 OXE 大规模预训练的 0.70；在空间更复杂的 Click Bell 上甚至超越 OXE 基线（0.35 vs 0.20）。这证明 FiT3D 的空间质量可部分弥补数据规模差距。

**最强证据：** Hard 设置下的全面领先，以及 FiT3D 教师相比 VGGT 的压倒性优势，直接证明了“编码器级对齐”与“同质 patch 特征教师”的关键性。

**最存疑证据：** 真实世界实验仅包含四项任务，每项 20 次试验，统计样本量有限；且任务难度相对温和，对极端三维精度的考验不如工业级装配场景严苛。

#

# 局限性

1. **教师模型的域限制：** DINOv2-FiT3D 仅在室内场景（ScanNet++）上微调，其在高度非结构化、室外或分布外环境中的空间迁移性尚未验证。
2. **真实世界验证规模有限：** 仅四项任务、每项 20 次试验，难以支撑强统计结论；且未涉及更复杂的 contact-rich 操作或动态环境。
3. **无法解决所有空间误差：** 失败案例分析显示，细粒度部件定位（如笔记本铰链侧识别错误）、抓取姿态传播误差、工作空间边界碰撞等问题依然存在，说明编码器级对齐不能替代显式的抓取规划或闭环修正。
4. **缺乏理论量化分析：** 论文主要通过实验归纳说明编码器级对齐优于 LLM 级对齐，但未提供如互信息、特征纠缠度等定量指标来严格刻画“语义污染”的程度。
5. **架构特定性：** 当前实现严格绑定 DINOv2 分支；对于采用单一视觉编码器或不同特征融合策略的 VLA 架构（如 $\pi_0$、GR00T），需重新验证对齐位置与投影器设计。

#

# 个人研究判断

面向“World Models assisting Embodied AI downstream tasks”这一研究方向，**VEGA 值得继续追踪**。

理由如下：

- **范式价值：** VEGA 提供了一种将三维世界模型（FiT3D 本质上是基于 3D Gaussian Splatting 的世界模型蒸馏）轻量注入 VLA 的有效路径。它证明了“在视觉入口层注入几何先验”比“在语言层找回几何”更干净、更高效，这对后续世界模型与具身智能的结合具有重要启发。
- **迁移潜力：** 该框架不局限于 FiT3D，未来可替换为更强的三维世界模型（如视频预测模型、3D 扩散模型、神经辐射场特征场等），且可迁移至其他 VLA 架构，通用性强。
- **效率优势：** 训练与数据效率的显著提升，对机器人学习这种高采样成本领域尤为珍贵，说明空间先验作为归纳偏置的实际工程价值。
- **明确延伸方向：** 论文局限也指出了下一步研究点——构建更通用、跨域的三维教师模型，将编码器级对齐与闭环控制、显式几何规划结合，以及在更多真实平台（如移动操作、人形机器人）上验证。这些方向正是世界模型辅助具身智能的核心议题。

综上，VEGA 是一篇在问题诊断、方法简洁性与实验完整性上均表现扎实的工作，为 VLA 的空间感知增强提供了可立即落地的新基线。
