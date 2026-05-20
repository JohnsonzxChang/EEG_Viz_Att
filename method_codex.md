# EEG + Tobii 眼动的主观图像注意读出范式与 SOP

版本日期：2026-05-05

## 1. 研究目标

本实验目标是利用 EEG 与眼动控制，在多物体自然图像刺激中读出人类主观视觉注意分配，即 attention map。关键区别不是从 EEG 中解码图像类别、物体身份或物体空间位置，而是在物理图像输入保持不变的条件下，仅通过事先视觉注意指示改变被试的主观任务相关注意，并检验 EEG 是否能读出这种 cue-driven attention map。

核心主张应表述为：

> 在相同多物体图像、相同物体布局、相同低级视觉统计的条件下，事先语义注意指示诱发不同主观注意分配；EEG 频率标记信号可读出被 cue 物体的神经增益。眼动数据用于确认中央注视和排除显性扫视，避免将结果解释为眼动或物体位置效应。

不建议使用“attention map 与图像物体位置完全无关”这种绝对表述。更严谨的表述是：

> 解码结果不能由物体位置、物理显著性、低级图像统计或眼动偏移解释，而反映事先指示驱动的主观任务相关注意。

## 2. 实验范式总览

### 2.1 基本设计

采用 `cue + 多物体同图 + 频率标记 + 固定注视 + 延迟反应` 的范式。

每张基础图像包含 4 个可辨认物体。每张图像在实验中重复 4 次，每次 cue 不同物体。关键是同一张图像在 4 个条件下的物理视觉输入完全相同，只有 trial 前的注意指示不同。

示例：

- 基础图像 A：杯子、钥匙、手机、笔记本。
- Trial A1：cue “杯子”。
- Trial A2：cue “钥匙”。
- Trial A3：cue “手机”。
- Trial A4：cue “笔记本”。

图像本身不变，cue 改变被试应关注的目标物体。

### 2.2 频率标记

每个物体区域叠加轻微正弦调制，作为 SSVEP/frequency tagging 标记。建议频率：

- 13 Hz
- 15 Hz
- 17 Hz
- 19 Hz

或根据显示器刷新率选择可稳定呈现的整数帧周期频率。若使用 120 Hz 显示器，应优先选取能用帧序列稳定近似或精确实现的频率，并通过光电二极管验证真实输出频率。

调制方式：

- 只调制物体 mask 内的亮度或透明度。
- 调制深度建议 8-15%，以能诱发 SSVEP 但不明显破坏图像自然性为准。
- 4 个物体同时调制，但频率不同。
- 频率与物体身份、物体位置、cue target 必须完全 counterbalance。

### 2.3 注意指示

cue 必须指向物体身份或语义属性，而不是空间位置。

推荐 cue：

- 中央文字：例如“杯子”“钥匙”。
- 中央小图标：如果图标不会引入位置线索。
- 语义类别：例如“可书写物”“可饮用容器”，用于更高阶注意任务。

避免 cue：

- 箭头。
- 左/右/上/下文字。
- 高亮框。
- 任何显式空间提示。

### 2.4 眼动角色

Tobii Eye Tracker 5 在本实验中用于：

- 判断 trial 开始前是否稳定中央注视。
- 记录刺激期间 gaze 是否偏离。
- 剔除眨眼、扫视和显性凝视目标物体的 trial。
- 将 gaze deviation 作为统计模型协变量。

不建议将 Tobii Eye Tracker 5 作为高精度 microsaccade 分析设备。其定位是注视质量控制和显性眼动排除，而不是作为主要神经指标。

## 3. 被试

### 3.1 样本量

最低建议：

- N = 30 名有效被试。

更稳健建议：

- N = 40-50 名有效被试。

考虑到 EEG、眼动、行为准确率和 trial 剔除，建议预招募比目标有效样本多 10-20%。

### 3.2 纳入标准

- 年龄 18-35 岁，或按研究目标扩展。
- 正常或矫正正常视力。
- 正常色觉。
- 能理解并完成注意任务。
- EEG 电极阻抗可控制在实验室标准范围内。

### 3.3 排除标准

- 光敏性癫痫史。
- 神经系统或严重精神疾病史。
- 视觉障碍或严重色觉异常。
- 眼动校准失败。
- EEG 噪声过大，导致有效 trial 数不足。
- 行为准确率低于预设阈值。

## 4. 刺激材料

### 4.1 图像要求

每张图像包含 4 个清晰可辨认物体，并满足：

- 物体之间不重叠或少量重叠。
- 物体分布在 4 个大致等距位置。
- 物体大小接近，避免某一物体显著更大。
- 物体亮度、对比度、颜色饱和度尽量平衡。
- 目标物体不应总在固定位置。
- 图像背景不应强烈吸引注意。

### 4.2 物体标注

每张图像需要为 4 个物体准备：

- 物体名称。
- 物体语义类别。
- 二值 segmentation mask。
- bounding box。
- 物体中心坐标。
- 物体面积。
- mean luminance。
- contrast。
- saliency score。

mask 可由人工标注、SAM/SAM2 辅助标注或现成数据集标注获得。最终必须人工检查。

### 4.3 低级特征控制

对每个物体计算并记录：

- 面积。
- eccentricity。
- 平均亮度。
- RMS contrast。
- 色彩饱和度。
- 空间频率能量。
- saliency model 分数。

这些变量不一定全部严格匹配，但必须在统计模型中作为协变量，并在刺激选择阶段避免极端不平衡。

### 4.4 刺激数量

最低可行版本：

- 160 张基础图像。
- 每张图像 4 个 cue 条件。
- 共 640 trials/被试。

推荐版本：

- 240 张基础图像。
- 每张图像 4 个 cue 条件。
- 共 960 trials/被试。
- 分 2 个 session 完成。

相同基础图像的不同 cue 条件之间至少间隔 20 trials，避免被试看出重复结构或使用记忆策略。

## 5. 设备与同步

### 5.1 EEG

推荐配置：

- 64 通道或以上 EEG。
- 采样率 1000 Hz。
- 在线参考按设备标准设置，离线重参考为 average reference 或 mastoid reference。
- 重点关注 occipital 与 parieto-occipital 电极。

关键电极区域：

- O1/O2/Oz。
- PO3/PO4/PO7/PO8/POz。
- P3/P4/Pz。

### 5.2 显示器

推荐：

- 120 Hz 或 144 Hz 刷新率。
- 固定亮度与色彩模式。
- 关闭动态刷新率、HDR、自动亮度、护眼模式。
- 使用全屏独占或稳定呈现模式。

必须使用光电二极管验证：

- stimulus onset timing。
- flicker frequency。
- frame drop。
- trigger 与真实显示之间的延迟。

### 5.3 眼动

使用 Tobii Eye Tracker 5：

- 实验开始前完成 5 点或 9 点校准。
- 每 80-100 trials 重新校准或检查 drift。
- trial 开始前要求 gaze 位于中央 fixation window。
- 刺激期间记录 gaze 坐标、pupil、validity。

### 5.4 同步

推荐使用 Lab Streaming Layer 或等效方案同步：

- EEG trigger。
- stimulus onset。
- cue onset。
- image onset。
- response onset。
- Tobii gaze stream。
- photodiode channel。

每个 trial 至少记录以下事件码：

- fixation onset。
- cue onset。
- stimulus onset。
- mask onset。
- probe onset。
- response。
- feedback 或 trial end。

## 6. 单 trial 流程

### 6.1 时间结构

推荐 trial 时序：

1. Fixation：800 ms。
2. Cue：500 ms。
3. Cue-stimulus interval：500-800 ms jitter。
4. Multi-object image with frequency tagging：2000 ms。
5. Mask：300 ms。
6. Probe response：最长 2000 ms。
7. ITI：800-1200 ms jitter。

### 6.2 注视门控

trial 开始前：

- 被试必须注视中央 fixation。
- gaze 必须在中央 1 degree visual angle 内保持至少 300 ms。
- 若不满足，trial 不开始或延迟开始。

刺激呈现期间：

- 要求被试始终注视中央。
- 不允许看向目标物体。
- 如果 gaze 偏离中央超过 1.5 degree 且持续超过 100 ms，该 trial 标记为 gaze violation。

### 6.3 行为任务

刺激结束后呈现 probe。probe 只询问被 cue 物体相关细节，避免被试平均关注所有物体。

示例 probe：

- “刚才被提示的物体是否有把手？”
- “被提示物体的颜色更接近 A 还是 B？”
- “被提示物体是否发生了轻微朝向变化？”
- “哪个细节属于被提示物体？”

probe 设计原则：

- 问题只在刺激结束后出现。
- 反应阶段不进入主要 EEG 分析窗口。
- 行为准确率目标为 75-90%。
- 低于 65% 表明注意操控可能无效。
- 高于 95% 表明任务过易，应增加细节辨别难度。

## 7. 实验分块

推荐结构：

- 每 block 40-60 trials。
- 每 block 约 5-8 分钟。
- 每 block 后强制休息。
- 每 2 个 block 做一次眼动 drift check。
- 每 4 个 block 检查 EEG 阻抗和信号质量。

总时长：

- 640 trials：约 90-120 分钟，含准备。
- 960 trials：建议分两天或两个 session，每 session 90-120 分钟。

## 8. Counterbalancing

必须平衡以下因素：

- cue target：4 个物体每个都被 cue 一次。
- object position：目标物体在 4 个位置的概率相同。
- flicker frequency：每个物体身份、位置、cue target 分配到各频率的概率相同。
- response hand：按键左右平衡。
- probe answer：yes/no 或 A/B 平衡。
- image repetition distance：同一基础图不同 cue 条件不连续出现。

建议用 Latin square 或完全平衡表生成 trial sequence。

## 9. EEG 预处理

### 9.1 基础预处理

推荐流程：

1. 导入 EEG、trigger、photodiode、gaze 数据。
2. 根据 photodiode 校正 stimulus onset。
3. 滤波：
   - ERP 分析：0.1-40 Hz。
   - SSVEP 分析：保留目标频率及其邻近频段，可用 1-45 Hz 或更宽。
4. 去除坏通道。
5. ICA 或等效方法去除眨眼、水平眼动、肌电。
6. 重参考。
7. epoch：
   - cue-locked epoch。
   - stimulus-locked epoch。
8. baseline correction：
   - ERP 用 pre-cue 或 pre-stimulus baseline。
   - SSVEP 以刺激前频谱或邻近频率作为噪声估计。

### 9.2 Trial 剔除

剔除或标记：

- blink 超过 200 ms。
- gaze 偏离中央超过 1.5 degree 且持续超过 100 ms。
- EEG peak-to-peak amplitude 超过 120 uV。
- 目标电极噪声异常。
- response 过快或过慢。
- probe 错误 trial。

建议主分析只使用：

- gaze-valid trials。
- EEG-clean trials。
- behavior-correct trials。

probe 错误 trial 可作为补充分析，检验注意失败时 EEG attention gain 是否降低。

## 10. 主要 EEG 指标

### 10.1 SSVEP 注意增益

对刺激期 0.5-2.0 s 或 0.3-2.0 s 计算每个标记频率的功率或 SNR。

定义：

```text
logSNR(f) = log power(f) - mean(log power(neighbor frequencies))
AttentionGain = logSNR(target frequency) - mean(logSNR(non-target frequencies))
```

neighbor frequencies 应避开其他标记频率及其谐波。

主假设：

```text
AttentionGain > 0
```

即被 cue 物体对应频率的 SSVEP SNR 高于未 cue 物体频率。

### 10.2 ERP 与 alpha 作为补充指标

可补充分析：

- cue-locked alpha lateralization。
- stimulus-locked N2pc。
- CDA/SPCN。
- posterior alpha suppression。
- time-resolved decoding。

但主指标建议预注册为 SSVEP frequency tagging，因为它最直接对应多物体并行 attention readout。

## 11. Attention map 构建

每张图像有 4 个物体 mask。对每个 trial 或条件，EEG 给出 4 个物体的注意权重：

```text
w_i = normalized logSNR(f_i)
```

将权重映射回图像：

```text
AttentionMap(x, y) = sum_i w_i * Mask_i(x, y)
```

可选平滑：

- 在物体 mask 内保持权重常数。
- 对 mask 边界做轻微 Gaussian smoothing。
- 不要让平滑扩散到未标注背景过多区域。

注意：

- map 的空间轮廓来自预先定义的 object mask。
- map 的强度来自 EEG。
- 不能把 object mask 本身当作 EEG 解码结果。

## 12. 统计分析

### 12.1 主模型

推荐线性混合模型：

```text
AttentionGain ~ CueTarget + TargetPosition + Frequency + Saliency
              + ObjectArea + Luminance + Contrast + GazeDeviation
              + (1 | Subject) + (1 | Image)
```

关键检验：

- cue target 对目标频率增益是否显著。
- target position 是否不能解释主效应。
- saliency、area、luminance、contrast 是否不能解释主效应。
- gaze deviation 加入模型后主效应是否仍存在。

### 12.2 解码验证

建议至少做 3 个 cross-validation：

1. Leave-image-out：
   - 训练图像与测试图像不重叠。
   - 排除对具体图片的记忆或低级统计捷径。

2. Leave-position-out：
   - 训练某些位置，测试未见位置。
   - 检验读出是否不依赖固定空间位置。

3. Leave-frequency-assignment-out：
   - 改变频率与物体/位置的绑定。
   - 排除分类器只学到某个频率或频率噪声模式。

最关键的验证：

```text
同一张基础图像，不同 cue 条件下，EEG attention map 可区分被试主观关注的物体。
```

### 12.3 眼动控制分析

必须报告：

- 各 cue 条件 gaze x/y 是否有系统差异。
- 目标物体方向的 gaze bias 是否显著。
- 剔除 gaze bias 最大的 trial 后结果是否仍成立。
- 仅使用严格中央注视 trial 时主效应是否保留。

## 13. 质量控制

### 13.1 实验前

检查：

- 显示器刷新率固定。
- flicker 频率通过光电二极管验证。
- EEG trigger 与 photodiode 同步。
- Tobii 校准误差可接受。
- 被试理解不能看向目标物体。

### 13.2 实验中

每个 block 后检查：

- EEG 噪声。
- 坏通道。
- gaze validity。
- 行为准确率。
- 被试疲劳。

若行为准确率连续两个 block 低于 65%，需要暂停并重新说明任务。

### 13.3 实验后

确认：

- 每个条件有效 trial 数足够。
- 每个频率有效 trial 数平衡。
- 每个位置有效 trial 数平衡。
- 眼动剔除没有导致某一 cue 条件 trial 大量损失。

## 14. 预注册建议

建议预注册以下内容：

- 主假设：cue target frequency 的 SSVEP SNR 高于 non-target frequencies。
- 主分析窗口：例如 stimulus onset 后 0.5-2.0 s。
- 主 ROI：occipital/parieto-occipital 电极，最好由独立 localizer 定义。
- trial 剔除标准。
- gaze exclusion 标准。
- 行为准确率纳入标准。
- mixed model 公式。
- leave-image-out、leave-position-out、leave-frequency-assignment-out 验证。

## 15. 与主流图片-EEG工作的差异化

主流图片-EEG研究通常回答：

- EEG 能否解码图像类别。
- EEG 能否重建图像语义或视觉表征。
- EEG 表征与 CNN/CLIP/ViT 表征是否相似。
- EEG 解码是否受低级图像统计影响。

本范式回答：

- 在图像完全相同的情况下，cue 改变的主观注意状态能否被 EEG 读出。
- EEG 读出的 attention map 是否独立于物体位置、物理显著性和眼动偏移。
- 多物体自然图像中的 object-based attention 是否可以转化为 trial-level 或 condition-level attention map。

因此，论文定位不应是“EEG image decoding”，而应是：

```text
EEG readout of subjective object-based attention maps in identical multi-object natural images.
```

## 16. 最小可执行版本

如果先做 pilot，建议：

- 10 名被试。
- 80 张基础图像。
- 4 cue 条件。
- 320 trials/被试。
- 4 个 SSVEP 频率。
- 严格中央注视。
- 主指标只看 AttentionGain。

pilot 成功标准：

- 行为准确率 75-90%。
- gaze violation trial < 25%。
- occipital ROI 中目标频率 AttentionGain 显著大于 0。
- 同一图不同 cue 的目标频率增强方向一致。

## 17. 完整实验版本

正式实验建议：

- 40-50 名有效被试。
- 240 张基础图像。
- 960 trials/被试。
- 2 session。
- EEG + Tobii + photodiode 完整同步。
- 预注册主分析。
- 同时报告 SSVEP、眼动控制、行为表现和 attention map reconstruction。

## 18. 关键风险与规避

风险 1：被试直接看向目标物体。

规避：

- 中央注视门控。
- gaze violation 剔除。
- gaze deviation 作为协变量。
- 严格中央注视子集复现主结果。

风险 2：分类器读出物体位置。

规避：

- cue 不提供空间信息。
- 位置完全 counterbalance。
- leave-position-out 验证。
- 统计模型加入 target position。

风险 3：分类器读出低级图像统计。

规避：

- 同一基础图像不同 cue 条件物理输入相同。
- 低级特征作为协变量。
- leave-image-out 验证。
- 刺激选择时平衡 saliency、area、luminance、contrast。

风险 4：频率本身造成偏差。

规避：

- 频率与物体、位置、cue target 完全平衡。
- leave-frequency-assignment-out 验证。
- 光电二极管确认频率稳定。

风险 5：任务过难或过易。

规避：

- pilot 调整 probe 难度。
- 保持准确率 75-90%。
- 错误 trial 单独分析。

## 19. 建议报告的主要结果图

1. 范式示意图：cue、fixation、多物体频率标记、probe。
2. 同一图像不同 cue 条件示意。
3. 4 个频率的 occipital SSVEP spectrum。
4. target vs non-target logSNR。
5. AttentionGain 时间窗结果。
6. gaze heatmap，证明中央注视。
7. EEG-derived object attention map。
8. leave-image-out / leave-position-out decoding performance。

## 20. 方法学一句话摘要

通过在同一多物体自然图像中对不同物体进行频率标记，并用非空间语义 cue 操控被试的主观关注对象，本范式在严格中央注视和眼动协变量控制下，用 EEG SSVEP 注意增益重建 object-level attention map，从而区分主观注意读出与传统图像/位置解码。
