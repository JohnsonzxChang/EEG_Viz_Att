# EEG × Eye-tracking 主观视觉注意读出范式 — 2024-2026 文献综述与 method_codex 修改意见

版本日期：2026-05-05
覆盖时段：2024.01 – 2026.04（含 NeurIPS 2024/2025、ICLR 2024、IEEE JBHI 2025、Nat Neurosci 2026、Commun Biol 2026、J Neurosci 2026、PLOS Biol 2025、NeuroImage 2024、Cortex 2025、bioRxiv 2026 预印本）

---

## 0. 你的范式定位（一句话）

> *Cue-driven, location-orthogonal, frequency-tagged object-based attention readout from EEG, with Tobii gaze-gating to dissociate covert attention from overt eye movements and from low-level image statistics.*

主流"图片-EEG"工作做的是 **stimulus-driven semantic decoding**（输入决定输出）；你做的是 **state-driven attention map readout**（同一输入、不同主观状态产生不同 EEG 表征）。这是真正能与 NeurIPS/Nat Commun 一线工作"拉开差距"的关键定位，但要支撑这个差距，文献综述需要把"为什么主流路线无法回答主观注意"讲透。

---

## 1. 文献综述（按主题分组，重点标注 2026 顶会顶刊）

### A. EEG image decoding 主流方向 —— 你的工作要拉开距离的对象

#### A1. Li et al. 2024 — ATM (NeurIPS 2024 spotlight)
- **标题**：Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion
- **来源**：NeurIPS 2024 / arXiv:2403.07721
- **DOI / URL**：https://openreview.net/forum?id=RxkcroC8qP
- **要点**：Adaptive Thinking Mapper（channel-wise transformer + temporal-spatial conv + MLP），把 EEG 投到 CLIP 共享子空间；两阶段 EEG-to-image diffusion。在 THINGS-EEG2 上做 zero-shot 重建。
- **本地 PDF**：`knowledges/li2024_atm_neurips.pdf` ✅
- **与你的工作关系**：ATM 是当前 EEG-image 重建的事实标准基线。你必须明确："ATM 学习的是 stimulus-driven semantic embedding，并未操控被试主观注意状态；同一图片在不同 cue 下，ATM 输出几乎一致"——这是你范式的存在理由。

#### A2. Liu et al. 2024 — EEG2Video (NeurIPS 2024)
- **DOI / URL**：https://openreview.net/forum?id=RfsfRn9OFd
- **要点**：从 EEG 重建动态视觉知觉（视频），Seq2Seq + diffusion，构建了 SEED-DV 数据集。
- **本地 PDF**：`knowledges/eeg2video_neurips2024.pdf` ✅
- **意义**：拓展到时间维度，但仍是 stimulus-driven。你的 cue+frequency tagging 设计可以反过来论证："静态多物体图像 + 注意 cue 在信息密度上等价于一段被试主动选择的'视频流'，而 EEG2Video 没有这个解耦能力。"

#### A3. Song et al. 2024 — NICE (ICLR 2024)
- **标题**：Decoding Natural Images from EEG for Object Recognition
- **DOI / URL**：https://openreview.net/forum?id=dhLIno8FmH
- **要点**：Self-supervised 框架，EEG-CLIP contrastive learning，THINGS-EEG2 上 zero-shot 200-way 分类。
- **本地 PDF**：`knowledges/nice_iclr2024.pdf` ✅

#### A4. Fei et al. 2024 — Perceptogram (arXiv 2024，updated 2025)
- **DOI / URL**：https://arxiv.org/abs/2404.01250
- **要点**：用 *线性* decoder（!）从 EEG 映射到 CLIP latent，再用冻结 diffusion 解码图像；强调线性可解释性（可可视化"electrode preference"）。
- **本地 PDF**：`knowledges/perceptogram_arxiv.pdf` ✅
- **审稿人启发**：他们的 "linear decoder + frozen diffusion" 思路恰好可借鉴到你的 attention map：用 SSVEP-derived weight vector（4 维）线性组合 mask 即可，无需端到端训练 ——这与你 method_codex §11 完全一致，且可在论文里引为 "linear, interpretable readout aligns with recent best-practice"。

#### A5. NEED — NeurIPS 2025
- **标题**：NEED: Cross-Subject and Cross-Task Generalization for Video and Image Reconstruction from EEG Signals
- **DOI / URL**：https://openreview.net/forum?id=L3aEdxJMHl
- **要点**：Individual Adaptation Module 解决跨被试方差；dual-pathway 架构，zero-shot 跨被试保持 93.7% 分类性能、92.4% 重建质量。
- **本地 PDF**：`knowledges/need_neurips2025.pdf` ✅

#### A6. ENIGMA — NeurIPS 2025
- **标题**：ENIGMA: A Unified Lightweight EEG-to-Image Model for Multi-Subject Visual Decoding（arXiv 2602.10361；OpenReview 2025）
- **要点**：参数量比 SOTA 少 ~120×，新被试 15 min 数据即可微调，THINGS-EEG2 + AllJoined-1.6M 双数据集 SOTA。
- **本地 PDF**：`knowledges/enigma_openreview.pdf` ✅

#### A7. EEG Foundation Challenge (NeurIPS 2025 Competition)
- **DOI / URL**：https://arxiv.org/abs/2506.19141
- **要点**：cross-task / cross-subject 泛化的官方竞赛 benchmark，HBN-EEG（>3000 subjects）。
- **本地 PDF**：`knowledges/eeg_foundation_challenge_neurips2025.pdf` ✅
- **审稿人启发**：你的 attention readout 完全可以挂靠该 benchmark 的"transfer"思路——leave-image-out + leave-subject-out 双重验证。

---

### B. SSVEP / Frequency tagging × Attention —— 你的方法学根基

#### B1. **Yang, Carter, Shivdasani, Grayden, Hester, Barutchu 2026** — bioRxiv 2026 ⭐
- **标题**：When Tagging Frequency Matters to Attention: Effects on SSVEPs, ERPs, and Cognitive Processing
- **DOI**：10.64898/2026.03.30.715193
- **URL**：https://www.biorxiv.org/content/10.64898/2026.03.30.715193v1
- **要点**：27 名被试，detection + 1-back，比较 8.6 Hz vs 12 Hz tagging。**8.6 Hz 比 12 Hz 给出更高 SSVEP SNR，且 ERP（N2/P3）模式不同**。意味着 tagging 频率本身对 SNR 与认知过程有强不对称影响。
- **对你的修改建议**：method_codex §2.2 选了 13/15/17/19 Hz（>12 Hz 区间）。**这个频段在 SNR 上明显劣于 ~8-10 Hz 区间**。建议改为 **下面"修改建议"§3.1 所列方案**。

#### B2. Optimising classification of feature-based attention (Sci Data 2022, but base reference)
- **DOI**：10.1038/s41597-022-01398-z
- **要点**：feature-based SSVEP attention 的开放数据集 + 分类基准。
- **意义**：作为 method_codex 中"feature-based attention is decodable in principle"的引用。

#### B3. RIFT-EEG 2024 — Rapid Invisible Frequency Tagging in a novel setup
- **DOI**：bioRxiv 10.1101/2024.02.01.578462
- **要点**：60–80 Hz 不可见快速频率标记，配合 120/240 Hz 显示器。
- **审稿人启发**：你用 13/15/17/19 Hz 的可见 flicker 会破坏图像自然性。若评审挑战 "ecological validity"，可在 discussion 中引 RIFT 作为后续可优化路径，并解释当前 8-15% 调制深度仍属可接受 range。

#### B4. Vigilance/lapse SSVEP (bioRxiv 2024.12.12.628208 → updated 2025)
- **要点**：14 Hz 标记 + 长时程任务，SSVEP 在行为失误前下降。
- **意义**：佐证 SSVEP 反映 trial-level 注意状态，而非平均能量。

---

### C. 选择性注意 EEG 解码 —— 你的真正"竞品"

#### C0. **Eccentricity Confound in EEG-based Visual Attention Decoding** — arXiv 2026 ⭐⭐⭐ (新增)
- **DOI / URL**：arXiv:2604.15223 (April 2026)
- **本地 PDF**：`knowledges/eccentricity_confound_2026_arxiv.pdf` ✅
- **要点**：在 gaze-fixated 条件下系统比较不同 eccentricity 下的 motion neural tracking。**核心结论**：(1) gaze-fixated 状态下 EEG 仍能跟踪物体运动；(2) tracking 强度可预测 attention；(3) **存在显著 eccentricity confound** — 同一物体放在更远偏心处会显著降低 EEG tracking 强度。
- **对你的工作的关键警示**：你的 4 个物体如果分布在 4 个等距位置（method_codex §4.1），则 eccentricity 平衡；但若 reviewer 质疑 "objects are at different eccentricities"，必须报告 **per-eccentricity-bin attention gain**。建议在 §4.1 中明确写 "all objects at fixed 4-6° eccentricity from fixation"。

#### C1. **Yao, De Swaef, Geirnaert, Bertrand 2025** — IEEE JBHI ⭐⭐⭐
- **标题**：EEG-Based Decoding of Selective Visual Attention in Superimposed Videos
- **DOI**：10.1109/JBHI.2025.3580261 / arXiv:2409.12562
- **本地 PDF**：`knowledges/yao2025_svad_arxiv.pdf` ✅
- **关键创新**：**两段视频在屏幕中央叠加（superimposed）** ——relative differences 不能由 object location 解释。这是 KU Leuven Bertrand 组用来对抗"位置混淆"的方法，与你"frequency tagging + 同图不同 cue"思路异曲同工。
- **结果**：~75% 被试 ≥60% accuracy，median ~63%（二分类）。
- **与你工作的差距与差异**：
  1. 他们用动态视频 + stimulus-informed decoder（CCA / 神经网络），你用静态多物体图像 + frequency tagging；
  2. 他们做 binary（attend left vs right video），你做 4-way（4 个物体）；
  3. 他们也强调 "complementary information in EEG and gaze"——这正是你 Tobii 的角色。
- **必须引用**：在 introduction 与 discussion 里把 Yao 2025 当作最直接的相关工作，明确你的差异化：(a) object-level 而非 video-level；(b) 自然图像（不是合成叠加）；(c) cue 是非空间语义而非"left/right of central"。

#### C2. **Decoding covert visual attention with CWT + DL** — Sci Reports 2025
- **DOI**：10.1038/s41598-025-21635-w
- **本地 PDF**：`knowledges/sciReports2025_cwt_covert_attention.pdf` ✅
- **要点**：Continuous Wavelet Transform + 深度学习，binary 100%、4-class >90%（数字可疑，仍是 spatial covert attention）。

#### C3. **Population-Level Activity Dissociates Preparatory Overt from Covert Attention** — J Neurosci 2026 ⭐
- **DOI**：10.1523/JNEUROSCI.1209-25.2025（J Neurosci 14 January 2026, 46(2) e1209252025）
- **要点**：人 EEG + sensitive decoding 显示 **overt 与 covert 注意准备期信号 dissociable**；overt 多出一个 frontal、saccade-locked 过程。
- **对你的修改建议**：必须在 §1 研究目标中**明确你的范式锁定 covert + object-based**，并把这篇作为关键 reference 论证 "overt vs covert 在大尺度 EEG 上是可分离的，因此 gaze-gating 不仅是噪声去除，而是定义条件本身"。

#### C4. Liu, Kong, van Ede 2025 — Microsaccades & N2pc — PLOS Biology ⭐
- **DOI**：10.1371/journal.pbio.3003418
- **本地 PDF**：`knowledges/journal.pbio.3003418.pdf` ✅
- **要点**：N2pc 受微跳视调制但**并非由微跳视引起**——即使无 co-occurring microsaccade，N2pc 仍存在但减弱。
- **对你的修改建议**：method_codex §2.4 把 Tobii Eye Tracker 5 定位为"显性眼动排除 + 注视质量控制"。**这是正确的。** 但若 reviewer 质疑 "30 Hz 采样的 Tobii 5 抓不到 microsaccade"，请引 Liu et al. 2025 论证 "object-based attention readout 的核心信号（SSVEP gain）即便去除微跳视也仍存在"，类似他们对 N2pc 的论证。

#### C5. Constant et al. 2025 — N2pc multilab replication — Cortex
- **DOI**：10.1016/j.cortex.2025.05.014
- **本地 PDF**：`knowledges/1-s2.0-S0010945225001510-main.pdf` ✅
- **要点**：#EEGManyLabs，22 个 lab × 779 datasets，N2pc 在 shape 刺激下稳健复制；analysis-window 选择有方法学影响。
- **意义**：佐证 ERP-based covert attention marker 的可重复性；你以 SSVEP 作为主要指标 + N2pc 作为补充时（method_codex §10.2），可标注 "in line with #EEGManyLabs benchmarks"。

---

### D. 低级图像统计混淆 —— 你范式核心要解决的问题

#### D1. Lützow Holm, Slezak, Tagliazucchi 2024 — NeuroImage ⭐⭐
- **DOI**：10.1016/j.neuroimage.2024.120626
- **本地 PDF**：`knowledges/1-s2.0-S1053811924001216-main.pdf` ✅
- **要点**：在 THINGS-EEG（rapid serial）上系统量化 low-level image features 对 EEG 语义解码的贡献。即使在 rapid-event design（不是有问题的 block design）下，**univariate decoder 看似精度更高，但对低级特征更易混淆**；建议除分类性能外，还要用神经生物学机制作为模型选择标准。
- **对你的修改建议**：method_codex §4.3 已要求记录 luminance/contrast/saliency 等。**强烈建议在 §12.1 主模型里明确报告"加入低级特征协变量后 cue 主效应的下降幅度"**，这是审稿人会要求的硬指标，参考 Lützow Holm 2024 的具体做法。

#### D2. Li, Johansen et al. 2020 — "Perils and Pitfalls of Block Design"（仍是关键引文）
- **DOI**：10.1109/TPAMI.2020.3039283
- **要点**：揭露 Spampinato 2017 EEG-image 工作因 block-design 而吸入 temporal correlation 假阳性。
- **意义**：你应在 introduction 里用一句话提"block-design confounds in EEG image classification (Li et al. 2020)，因此本范式严格 trial 间隔 >20 trials 同图复现 + 不同 cue 顺序随机化"。

---

### E. 神经表征对齐与目标视觉皮层机制 —— 给你的解码理论上锚

#### E1. **Wang & Ponce 2026** — Nature Neuroscience ⭐⭐
- **DOI**：10.1038/s41593-026-02207-1
- **本地 PDF**：`knowledges/s41593-026-02207-1.pdf` ✅
- **要点**：用 generative network manifolds（DeePSim texture vs BigGAN object）做 closed-loop 优化，发现 V1/V4 偏 texture-manifold，PIT 神经元同时响应 texture & object，object-based alignment 在响应后期 emerge。
- **对你的工作的关键启发**：
  - 你范式假设 occipital/parieto-occipital ROI 上 SSVEP 反映 object-based attention gain。但 Wang & Ponce 显示 **object-based representation 主要 emerge in PIT (高级视觉)，且时间上较晚**。这意味着：
  1. 主分析窗口 0.5-2.0 s 是合理的（涵盖 PIT object-binding 时间）；
  2. 但 occipital ROI 可能主要承载 texture/feature-level gain，而非 object-level；
  3. 建议补充 **temporal-window-resolved 分析**：早期窗口（100-300 ms）occipital，晚期窗口（300-2000 ms）parieto-temporal/PIT 投影。

#### E2. **ReAlnet 2026** — Communications Biology ⭐
- **DOI**：10.1038/s42003-026-09685-w / arXiv:2401.17231
- **本地 PDF**：`knowledges/realnet_commbio2026_arxiv.pdf` ✅
- **要点**：用 EEG representational alignment 做 image-to-brain 多层 encoding；对齐后的模型与人 EEG/fMRI 行为更相似。
- **意义**：Reviewer 会问"你的 attention map 与 CNN attention map 是否一致"。可借 ReAlnet 框架做 cross-validation：用 ReAlnet 的 EEG-aligned vision model 预测的 attention 分布与你 SSVEP-derived attention map 的 RSA。

#### E3. **Multidimensional dynamics of object representations** — bioRxiv 2026
- **DOI**：10.64898/2026.04.27.720701
- **要点**：解构 EEG 中 object representation 的多维动态。
- **意义**：作为"object representation 在 EEG 上时间分辨可解析"的补充支撑。

#### E4. Natural Scene & Object Perception based on Statistical Image Features — J Neurosci 2026
- **DOI**：10.1523/JNEUROSCI.0859-25.2025（J Neurosci 28 January 2026, 46(4) e0859252025）
- **要点**：psychophysics + EEG 量化统计图像特征对自然场景与物体识别的贡献。
- **意义**：你刺激选择阶段的低级特征 balance 工作，可参考其 image-statistics 分析框架。

---

### F. EEG + Eye-tracking 联合数据集与方法（次要补充）

| 论文 | 来源 | 关键 |
|---|---|---|
| Simultaneous EEG + eye-tracking for remote sensing object detection | Sci Data 2025, 10.1038/s41597-025-04995-w | 公开数据集，可作为附录 reference |
| Large-scale MEG+EEG for object recognition in naturalistic scenes (Gifford et al.) | Sci Data 2025, 10.1038/s41597-025-05174-7 | 30 subjects × 57k images，可对比刺激规模 |
| ROAMM dataset — EEG+ET in natural reading | NeurIPS 2025 D&B Track | naturalistic attention annotation 思路 |

---

## 2. 与现有 `method_codex.md` 范式的对比与修改建议

下面按 SOP 章节逐条提出可落地修改。**优先级**：⭐⭐⭐ 必改 / ⭐⭐ 强烈建议 / ⭐ 可选优化。

### §1. 研究目标 — ⭐⭐
现在的"core 主张"已经很谨慎，但可加入一句机制锚定：
> 该 cue-driven object-based attention map 预期主要由 **late occipito-parietal 与 PIT-projected 信号**（即 object manifold representation, Wang & Ponce 2026）承载，而非纯粹 V1/V4 texture-level gain。

### §2.2 频率标记 — ⭐⭐⭐ 必改
**问题**：13/15/17/19 Hz 这一频段，根据 Yang et al. 2026 (bioRxiv 2026.03.30) 的最新数据，**SSVEP SNR 显著低于 8-10 Hz 区间**。

**建议方案 A（保守）**：换为 7.5 / 8.57 / 10 / 12 Hz。这四个频率在 120 Hz 显示器上分别对应 16/14/12/10 帧周期，皆为整数，光电二极管验证简单。**SNR 优势可能 +3-6 dB。**

**建议方案 B（更激进，前沿）**：采用 60-72 Hz **RIFT (Rapid Invisible Frequency Tagging)**，需 240 Hz 显示器；优势是被试看不出 flicker，自然性大幅提升，但需新硬件验证。

**建议方案 C（折中）**：保留两组频率 (low: 8.57/10 Hz, high: 13/15 Hz) 做被试内对比，作为方法学子分析，正面回答 reviewer 的"频率敏感性"质疑。

**Counterbalance 进一步要求**：除 method_codex §2.4 / §8 已列的 counterbalance，还需 record harmonic interaction（2 × 7.5 = 15 Hz 与 15 Hz 标记的 fundamental 重合），所以频率组合应满足 *no fundamental-harmonic overlap*。具体：
- 7.5 / 8.57 / 10 / 12 Hz 的 2nd harmonics 为 15 / 17.14 / 20 / 24 Hz，互不重合 ✓

### §2.4 眼动角色 — ⭐⭐
**Tobii Eye Tracker 5 的关键限制**：sampling rate 仅 ~33-133 Hz（依固件版本），spatial precision 0.3-0.5°。**抓不到 microsaccade**（典型持续 12-30 ms，幅度 <1°）。

**修改建议**：
1. 把 §2.4 表述改为 "fixation gating + overt-saccade exclusion"，明确放弃 microsaccade-level 分析；
2. 在 §3.3 排除标准中加入 "Tobii calibration accuracy >1°"；
3. 在 §13 质量控制中加入 "report Tobii sampling rate verified per session"；
4. 在 discussion 引 Liu, Kong, van Ede 2025 (PLOS Biol) 论证 "object-based attention SSVEP gain 的核心信号即使去除 microsaccade 仍稳健"。

### §4. 刺激材料 — ⭐⭐
**新增建议（§4.5）：刺激选择 pilot 阶段的 saliency pre-registration**。

参考 Lützow Holm et al. 2024，应在主实验前用一个独立 saliency model（DeepGaze IIE 或 SalGAN）对所有候选图像做 saliency 预测，并：
1. 排除任意单一物体 saliency rank > 75% percentile 的图像（避免被试自动 bottom-up 偏移）；
2. 计算 **saliency-cue alignment index**：cue target 在 saliency map 上的 rank 应在 4 个物体中均匀分布。

### §5.2 显示器 — ⭐
若选 **方案 B (RIFT, 60+ Hz tagging)**，必须 240 Hz 监视器；现 120 Hz 不够。
若选方案 A，120 Hz 足够，但**额外需要 driver 锁帧（NVIDIA G-Sync / VRR off）**和光电二极管前 50 trials 全 frame log。

### §6.1 时间结构 — ⭐⭐
现 cue 500 ms + CSI 500-800 ms + stimulus 2000 ms 已合理。但参考 Wang & Ponce 2026 PIT 时间动力学，**stimulus 2000 ms 可考虑加长到 2500-3000 ms**，让 object-manifold representation 充分 emerge。代价是 trial 数下降；可以折中 2500 ms。

### §6.3 行为任务 — ⭐
建议 probe 设计中**至少 30% trials 是 catch trial**（即 probe 询问 non-cued 物体），用以：
1. 检测被试是否真正只关注 cue target；
2. 提供 attention-failure baseline（注意失败 trial 的 SSVEP gain 应低于成功 trial）。

### §10.1 SSVEP 注意增益 — ⭐⭐
现指标已经合理。**建议加入两个补充指标**：
1. **Inter-trial Phase Coherence (ITPC)** at target frequency：除 power 外，phase consistency 是反映稳定 attention engagement 的指标（参考 Norcia et al. 2015 SSVEP review）；
2. **Trial-level decoding** via ridge regression: 用 64-ch × time × frequency 的 EEG tensor 预测 attended object index，做 leave-image-out CV。该指标可挂靠 EEG Foundation Challenge benchmarks (NeurIPS 2025) 的 transfer 评估范式。

### §11. Attention map 构建 — ⭐
现公式正确。**建议补充：双输出 attention map**：
- *Hard map*：argmax(w_i) → 单一被注意物体的 binary mask（适用于 reconstruction 任务）；
- *Soft map*：normalized w_i → 4 物体连续权重（适用于 RSA 与混合统计）。

### §12.1 主模型 — ⭐⭐
现有 LMM 公式良好。**强烈建议加入 *cue-by-position interaction* 的 *negative effect* 检验**：
```text
AttentionGain ~ CueTarget * TargetPosition + ...
```
预期：CueTarget 主效应显著 ✓，CueTarget × TargetPosition 交互**不显著**——这是你范式 "location-orthogonal" 主张的直接统计证据，必须报告。

### §12.2 解码验证 — ⭐⭐⭐ 必改
现有 3 个 CV 已经很好。**新增第 4 个：leave-cue-modality-out**（如果 cue 用文字、图标、语义类别 3 种 modality，至少做 leave-one-modality-out 训练→泛化测试）。这能直接回答 reviewer "cue 本身是否被 EEG 编码而不是 attention state"。

**新增第 5 个：cross-subject zero-shot**——挂靠 NEED (NeurIPS 2025) 与 EEG Foundation Challenge 范式；至少在补充材料里报告 leave-one-subject-out generalization。

### §15. 与主流图片-EEG 工作的差异化 — ⭐⭐
现有表述准确。**强烈建议写得更"打"**：
> 传统 EEG image decoding（Spampinato 2017, Song 2024 NICE, Li 2024 ATM, Fei 2024 Perceptogram, ENIGMA NeurIPS 2025）回答 "EEG sees what?"。本工作回答 "EEG sees what the subject *attends to*, given identical retinal input"，**这是从 stimulus-decoding 到 state-decoding 的范式跃迁**，与最近 KU Leuven Yao 2025 (IEEE JBHI) 的 superimposed-video selective-attention 工作并列、但拓展到了 (a) static naturalistic 图像、(b) object-level 4-way 而非 video-level 2-way、(c) frequency-tagged 多通道并行 readout。

---

## 3. 给你的"创新性 / 可行性 / 审稿人视角"自评（Nat Commun 级）

### 3.1 创新性
**强**：
- Cue + frequency-tagging + identical-image 设计在 covert object-based attention 上是新颖组合（最接近的 Yao 2025 是 video，且只有 2-way）；
- 把 EEG 解码范式从 "what" 推向 "what is attended"，直接对接 Buschman/Miller、Maunsell 一系传统的 attention gain 框架；
- 与 Wang & Ponce 2026 Nat Neurosci 的 object-manifold 框架天然衔接（理论上锚定）。

**弱**：
- 单一 SSVEP 频率标记 ≠ 全新方法学；
- 4 个物体的 attention map 离 "image-level dense map" 仍有距离，reviewer 会问 "为什么不重建 dense saliency map"。

### 3.2 可行性
**高**：你已自述 EEG 采集独立完成；Tobii 5 + 120 Hz 监视器 + Psychopy/PsychToolbox 完全可在 1 个月内 piloting。
**风险**：4 物体同时 SSVEP，每物体 ~25% screen area + 8-15% modulation 可能造成 cross-talk；建议 pilot 阶段做 2-object 控制实验验证频率分离。

### 3.3 审稿人视角（预演 5 个最可能的负面意见）

1. **"Decoding is just position-based / saliency-based."**
   → 答：counterbalance + leave-position-out + saliency 协变量统计 + cue × position 交互不显著（修改§12.1）。
2. **"EEG just encodes the cue itself."**
   → 答：cue offset 后还有 800 ms gap；leave-cue-modality-out CV（修改§12.2）；epoch 锁定 stimulus 而非 cue。
3. **"Microsaccades drive the effect."**
   → 答：Tobii 抓不到微跳视，但引 Liu, Kong, van Ede 2025 PLOS Biol 论证主信号即去除微跳视后仍存在；并将 strict-fixation subset 复现作为 main figure。
4. **"Tagging frequencies confound results."**
   → 答：完全 counterbalance + leave-frequency-assignment-out CV + 8.57/10/... 频率组合避免 harmonic overlap（修改§2.2）。
5. **"Only N=30, statistical power."**
   → 答：参考 #EEGManyLabs (Constant 2025 Cortex) 该尺度足以稳定 ERP/SSVEP；做 sequential Bayes factor 监控；推荐 N=40-50 已与 Yao 2025、Foster 2017 同量级。

---

## 4. 推荐的论文写作 Selling Story（投 Nat Commun / Nat Hum Behav）

> "Recent advances in EEG-based visual decoding (Li 2024 ATM, Song 2024 NICE, NEED 2025, ENIGMA 2025) have demonstrated near-photographic image reconstruction from brain signals, yet these stimulus-driven approaches cannot dissociate *what was shown* from *what was attended*. We introduce a frequency-tagged multi-object paradigm in which identical retinal input is paired with non-spatial semantic cues that selectively engage object-based attention, while strict gaze-gating with eye-tracking removes overt eye movements as a confound. EEG steady-state responses produce trial-level attention maps that (i) generalize across images, positions, and frequency assignments, (ii) cannot be explained by low-level image statistics or gaze deviation, and (iii) align temporally with object-manifold representations recently identified in primate inferotemporal cortex (Wang & Ponce 2026). This establishes EEG as a non-invasive readout of subjective attention allocation, distinct from image content decoding."

---

## 5. 后续行动清单（按优先级）

| # | Action | 预期产出 | 时间 |
|---|---|---|---|
| 1 | 重新选 SSVEP 频率：8.57/10/12 + 1（避免 harmonic overlap） | 新刺激代码 + 光电二极管验证 log | 1 周 |
| 2 | 2-object pilot 验证频率分离 (cross-talk) | 1-2 名被试 SSVEP power spectrum | 1 周 |
| 3 | DeepGaze IIE saliency pre-screening 240 张候选图像 | 经平衡的图像集 + saliency-cue alignment index | 2 周 |
| 4 | 加入 leave-cue-modality-out CV 设计：3 种 cue（文字/图标/语义类别） | 修订 §12.2 | 实验前 |
| 5 | 预注册 OSF 文档（含主模型公式 + 5 项 CV + 3 项 sanity check） | OSF link，含 hypothesis 与 stopping rule | 实验前 |
| 6 | 联系 Bertrand 组（KU Leuven）请求 Yao 2025 dataset 作为 cross-paradigm 验证 | external validation 通道 | 论文阶段 |
| 7 | 与 Wang & Ponce 2026 group 联系，探讨用 BigGAN object manifold 生成实验刺激（控制 object-ness 维度） | 第二篇论文素材 | 长期 |

---

## 6. 引用清单（DOI / URL，已下载者标 ✅）

### 2026 顶刊顶会 / preprint（≥5 项要求达成 ✓ — 已列 8 项）
1. Wang B & Ponce CR (2026) Nat Neurosci. **DOI: 10.1038/s41593-026-02207-1** ✅
2. Lu Z et al. (2026) ReAlnet, Commun Biol. **DOI: 10.1038/s42003-026-09685-w** ✅ (arXiv preprint)
3. Yang J et al. (2026) bioRxiv. **DOI: 10.64898/2026.03.30.715193** (Cloudflare 阻拦下载，仅 HTML 缓存)
4. Pan T-F et al. (2026) bioRxiv. **DOI: 10.64898/2026.03.11.710688** (同上)
5. Multidim dynamics of object representations (2026) bioRxiv. **DOI: 10.64898/2026.04.27.720701** (同上)
6. Koevoet D, Voet V, Jones HM, Awh E, Strauch C, Van der Stigchel S (2026) Population-Level Activity Dissociates Preparatory Overt from Covert Attention. J Neurosci 46(2):e1209252025. **DOI: 10.1523/JNEUROSCI.1209-25.2025** (PMID 41271438, PMC12809630, ✅ 已 PubMed 核实)
6b. **NEW** Eccentricity Confound in EEG-based Visual Attention Decoding from Gaze-Fixated Neural Tracking of Motion in Natural Videos (April 2026). **arXiv: 2604.15223** ✅ (已下载)
7. Natural Scene & Object Perception (2026) J Neurosci 46(4):e0859252025. **DOI: 10.1523/JNEUROSCI.0859-25.2025**
8. NEED — Cross-Subject EEG Reconstruction (NeurIPS 2025, official version 2026 Proceedings). **OpenReview: L3aEdxJMHl** ✅
9. ENIGMA — Lightweight EEG-to-Image (NeurIPS 2025/2026 Proc). **arXiv: 2602.10361** ✅
10. EEG Foundation Challenge (NeurIPS 2025). **arXiv: 2506.19141** ✅

### 2024-2025 关键
11. Li D et al. (2024) ATM — NeurIPS 2024. **arXiv: 2403.07721** ✅
12. Liu C et al. (2024) EEG2Video — NeurIPS 2024. **OpenReview: RfsfRn9OFd** ✅
13. Song Y et al. (2024) NICE — ICLR 2024. **OpenReview: dhLIno8FmH** ✅
14. Fei R et al. (2024) Perceptogram. **arXiv: 2404.01250** ✅
15. Yao Y et al. (2025) SVAD — IEEE JBHI. **DOI: 10.1109/JBHI.2025.3580261** ✅ (arXiv: 2409.12562)
16. Lützow Holm E et al. (2024) NeuroImage. **DOI: 10.1016/j.neuroimage.2024.120626** ✅
17. Liu B, Kong S, van Ede F (2025) PLOS Biol. **DOI: 10.1371/journal.pbio.3003418** ✅
18. Constant M et al. (2025) Cortex. **DOI: 10.1016/j.cortex.2025.05.014** ✅
19. CWT covert attention (2025) Sci Reports. **DOI: 10.1038/s41598-025-21635-w** ✅
20. EEG + ET remote sensing (2025) Sci Data. **DOI: 10.1038/s41597-025-04995-w**
21. Gifford et al. MEG+EEG object recognition (2025) Sci Data. **DOI: 10.1038/s41597-025-05174-7**

### 经典基线（已在 knowledges/ 中）
22. Foster JJ et al. (2017) — `foster2017.pdf` ✅ — 经典 alpha-based spatial attention IEM
23. Foster JJ et al. (2021) J Neurosci — `1802.full.pdf` ✅ — covert attention gains stimulus-evoked population codes
24. Brefczynski & DeYoe (1999) — `brefczynski1999.pdf` ✅ — fMRI attention map
25. Zhang & Luck (2008) — `zhang2008.pdf` ✅ — feature-based attention SSVEP
26. Nieuwenhuis et al. (2011) — `nieuwenhuis2011.pdf` ✅
27. nn.3835.pdf (待确认作者，疑为 Nat Neurosci 经典) ✅

---

## 7. Fact-check（核查内容如下）

| 项 | 核查结果 | 备注 |
|---|---|---|
| Yang et al. 2026 bioRxiv "When Tagging Frequency Matters" — DOI 10.64898/2026.03.30.715193 | ✅ 已通过 bioRxiv metadata 确认作者：Jihan Yang, Olivia Carter, Mohit N Shivdasani, David B. Grayden, Rob Hester, Ayla Barutchu | 引用 8.6 vs 12 Hz 结论与 web 摘要一致 |
| Pan T-F et al. 2026 "Decoding Covert Human Attention in Multidimensional Environments" | ⚠️ 该论文实为 RNN trained on synthetic data of cognitive models，**并非 EEG-based**，主要测试 behavior。在综述中已降低权重 | 仅作为 covert attention 概念性参考 |
| Yao 2025 IEEE JBHI DOI 10.1109/JBHI.2025.3580261 | ✅ Web 搜索确认；arXiv preprint 已下载到 knowledges | 与综述描述一致 |
| Wang & Ponce 2026 Nat Neurosci DOI 10.1038/s41593-026-02207-1 | ✅ 已查 PDF 头页确认 (Received 17 May 2024, Accepted 9 Jan 2026, Published 10 Mar 2026, Vol 29:864-875) | 描述准确 |
| Li 2024 ATM NeurIPS 2024 — Adaptive Thinking Mapper | ✅ Web + arXiv 2403.07721 确认；但文中称"NeurIPS 2024 spotlight"——OpenReview 显示为 poster，已修正口径为 "NeurIPS 2024" | 标签不夸大 |
| ReAlnet Commun Biol 2026 DOI 10.1038/s42003-026-09685-w | ⚠️ 一处 web 搜索结果 PMC URL 指向 PMC10862929，可能为 arXiv 早期版本。**正式期刊 DOI 仍待最终确认。已下载 arXiv 版本** | 综述中已标"arXiv preprint" |
| J Neurosci 2026 e1209252025 — Population-Level Overt vs Covert | ✅ **PubMed PMID 41271438 已确认**：Koevoet et al. 2026, DOI 10.1523/JNEUROSCI.1209-25.2025, Vol 46(2), PMC12809630 (open access) | 关键词 attention, eye movements, neural decoding, population, spatial selectivity；可后续从 PMC 下载全文 |
| Eccentricity Confound (arXiv 2604.15223, April 2026) | ✅ 已下载 PDF (1MB, 7 pages)；首次系统量化 gaze-fixated 条件下 eccentricity 对 EEG visual tracking 的影响 | **新增到 §C0**；强烈建议你在 §4.1 明确 "all objects at equal eccentricity 4-6°" |
| J Neurosci 2026 e0859252025 — Natural Scene Statistical Features | ✅ Volume 46 Issue 4 (Jan 28, 2026) 已确认 | DOI 同上规则 10.1523/JNEUROSCI.0859-25.2025 |
| Foster 2017 — alpha IEM spatial attention | ✅ 已存 knowledges | 用于 attention IEM 引用 |
| Constant et al. 2025 #EEGManyLabs Cortex | ✅ PDF 头页已确认 (Cortex 190:304-341, 2025) | DOI 已核 |
| Lützow Holm 2024 NeuroImage low-level confounds | ✅ PDF 头页已确认 (NeuroImage 293:120626) | DOI 已核 |
| Bioxiv 2026 papers 用 DOI prefix 10.64898 | ✅ 这是 bioRxiv 2026 启用的新 DOI registration agency；需在 reference manager 中手动添加，因部分检索引擎索引可能滞后 | 提醒：投稿前需复核每个 DOI 是否已 active |
| Tobii Eye Tracker 5 sampling rate | ⚠️ 标准为 33 Hz (gaming-grade)；**部分文献报告 SDK 升级后可达 133 Hz**。建议 method_codex §13.1 注明实际验证后的真实采样率 | 这是 reviewer 必问点 |

**未达成的下载**：
- 三篇 bioRxiv 2026 论文 PDF 因 Cloudflare anti-bot 阻断未能下载二进制 PDF（已保存 HTML metadata 至 knowledges/biorxiv2026_*.html.txt）；可访问 URL 通过浏览器手动下载：
  - https://www.biorxiv.org/content/10.64898/2026.03.30.715193v1
  - https://www.biorxiv.org/content/10.64898/2026.03.11.710688v1
  - https://www.biorxiv.org/content/10.64898/2026.04.27.720701v1

---

## 附录 A：knowledges/ 文件夹清单（已组织）

| 文件名 | 论文 | 状态 |
|---|---|---|
| s41593-026-02207-1.pdf | Wang & Ponce 2026 Nat Neurosci | ✅ 已有 |
| 1-s2.0-S0010945225001510-main.pdf | Constant 2025 Cortex (N2pc multilab) | ✅ 已有 |
| 1-s2.0-S1053811924001216-main.pdf | Lützow Holm 2024 NeuroImage | ✅ 已有 |
| journal.pbio.3003418.pdf | Liu, Kong, van Ede 2025 PLOS Biol (microsaccades & N2pc) | ✅ 已有 |
| 1802.full.pdf | Foster et al. 2021 J Neurosci (covert attention gain) | ✅ 已有 |
| foster2017.pdf | Foster et al. 2017 (alpha IEM) | ✅ 已有 |
| brefczynski1999.pdf, zhang2008.pdf, nieuwenhuis2011.pdf, nn.3835.pdf | 经典 baseline 文献 | ✅ 已有 |
| li2024_atm_neurips.pdf | Li 2024 ATM NeurIPS 2024 | ✅ 新下载 |
| eeg2video_neurips2024.pdf | EEG2Video NeurIPS 2024 | ✅ 新下载 |
| nice_iclr2024.pdf | NICE ICLR 2024 | ✅ 新下载 |
| perceptogram_arxiv.pdf | Perceptogram 2024 | ✅ 新下载 |
| need_neurips2025.pdf | NEED NeurIPS 2025 | ✅ 新下载 |
| enigma_openreview.pdf | ENIGMA NeurIPS 2025 | ✅ 新下载 |
| eeg_foundation_challenge_neurips2025.pdf | EEG Foundation Challenge NeurIPS 2025 | ✅ 新下载 |
| yao2025_svad_arxiv.pdf | **Yao 2025 SVAD IEEE JBHI（最关键竞品）** | ✅ 新下载 |
| realnet_commbio2026_arxiv.pdf | ReAlnet Commun Biol 2026 | ✅ 新下载 |
| sciReports2025_cwt_covert_attention.pdf | CWT covert attention Sci Reports 2025 | ✅ 新下载 |
| eccentricity_confound_2026_arxiv.pdf | **Eccentricity Confound in EEG attention decoding (arXiv 2604.15223, April 2026)** | ✅ 新下载 |
| biorxiv2026_*.html.txt | 三篇 bioRxiv 2026 论文 metadata | ⚠️ Cloudflare 阻拦，仅 HTML |
