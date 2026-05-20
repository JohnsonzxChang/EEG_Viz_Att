# EEG_Viz_Att — Stage Outputs

数据集：`data/zfn-0507/` — 1 subject, RSVP-attention LVIS paradigm, 6376 epochs × 34 channels × 1.5 s @ 1000 Hz.

每个 epoch 已 join 完成的元数据：
- `hint`  : 受试者注意类别 (40 classes)
- `image_id` + `targets_in_image` : COCO 图片 + 该图所含 K=2~4 目标
- `eeg_split` : train (2952) / test (3424) ── 范式预分好
- 所有 epochs 都是 `stim_category == hint` (即只保留 outlined target trial)

---

## Phase 1 (process1) — 数据 QC + ERP

`process1_data_process/` 包含：

| 文件 | 作用 |
|---|---|
| `data_io.py` | 加载 .fif + session JSON 双源数据，按 (image_id, outlined_label) 时序对齐，输出 `EpochBundle` (含 hint / split / repeat_index / targets / areas / is_target) |
| `qc_signal.py` | 每 epoch 的 RMS / 峰峰值 / kurtosis / flat-channel 计数 / GFP pre-post 比 / α-band SNR + 多标签共现矩阵 |
| `erp_viz.py` | grand-avg ERP (butterfly + GFP)、per-HINT ERP、超类聚合 ERP、train/test ERP 漂移检查、QC 分布直方图、类平衡条形图、多标签共现热力图 |
| `temporal_sweep.py` | 200ms 滑动窗 5-fold LDA on hint 标签，含 top-1 / balanced / top-5 |
| `run_qc.py` | 端到端入口，写 PNG + `process1_summary.json` |

### 运行命令
```bat
"C:\Users\thlab\.conda\envs\VIZ\python.exe" -m process1_data_process.run_qc ^
  --fif data\zfn-0507\epochs_big-epo.fif ^
  --session_json data\zfn-0507\session_rsvp_attention_lvis_pilot_20260507_201520.json ^
  --out_dir data\zfn-0507\process1_out ^
  --decim 4
```

### 已经在沙箱验证的事项
- 6376/7582 epochs 100% join 上 HINT/split/targets ── join 算法稳健
- 40 HINT 类别，最小 ~150、最大 ~205 trials → 类基本均衡
- train(2952) / test(3424) 分布与 session JSON 一致

### 待运行结果（缺 32 GB 内存的沙箱跑不了，需在用户机器上跑）
- ERP 图、QC 直方图、temporal sweep 准确率曲线 → `data/zfn-0507/process1_out/*.png`
- `process1_summary.json` 含 GFP_ratio / sweep peak / class balance

---

## Phase 2 (process2) — 神经网络框架 + 注意力预测对比

`process2_model_analysis/`：

| 文件 | 作用 |
|---|---|
| `models/eegnet.py` | EEGNet (Lawhence 2018) — 干净重写自 `old_analysis/encoder/eegnet_encoder.py`，提供 `forward_features` |
| `models/atm.py` | ATM Encoder (Li et al. NeurIPS 2024, arXiv:2403.07721) — Channel-Transformer + ShallowNet PatchEmbed + Residual MLP，干净重写自 `old_analysis/encoder/atm_encoder.py` |
| `models/fusion.py` | 三 baseline: `ImgOnlyClassifier` / `EEGOnlyClassifier` / `EEGImgFusionClassifier`（cross-attention 融合） |
| `img_embeddings.py` | 预计算 CLIP ViT-B/32 (默认) 或 ResNet-50 图片嵌入，缓存为 .npz |
| `dataset.py` | `EEGImgAttentionDataset` ── 把 EpochBundle + 嵌入缓存合成 PyTorch Dataset，自动 z-score |
| `compare_baselines.py` | 三 baseline 同 train/test split 训练 + 汇总三组 Δ 指标 |

### 运行序列
```bat
:: 1) 计算图片嵌入（一次性）
"C:\Users\thlab\.conda\envs\VIZ\python.exe" -m process2_model_analysis.img_embeddings ^
  --selection experiment\stimuli_select\stimuli_*.json ^
  --coco_root C:\Users\thlab\Desktop\ES_coco\data\coco ^
  --out_cache data\zfn-0507\img_embeddings_clip_vitb32.npz

:: 2) 三 baseline 训练 + 比较
"C:\Users\thlab\.conda\envs\VIZ\python.exe" -m process2_model_analysis.compare_baselines ^
  --fif data\zfn-0507\epochs_big-epo.fif ^
  --session_json data\zfn-0507\session_rsvp_attention_lvis_pilot_20260507_201520.json ^
  --img_cache data\zfn-0507\img_embeddings_clip_vitb32.npz ^
  --out_dir data\zfn-0507\process2_out ^
  --decim 4 --n_epochs 30 --batch_size 128 --device cuda
```

输出 `process2_out/compare_summary.json` 含核心结论字段：
- `delta_eeg_vs_img`     ── EEG-only 相对 img-only 的增益（**回答"EEG 是否提供独立于视觉先验的信息"**）
- `delta_fusion_vs_img`  ── EEG+Img 相对 img-only 的增益（**回答"加 EEG 后预测有多大提升"**）
- `delta_fusion_vs_eeg`  ── 验证 fusion 不止是"EEG 上加了一层" ── 检查 img 信息是否被有效利用

### 设计取舍
- 三个 baseline **故意** capacity 不对称 (img_only 最小)；如果 EEG-only 仍超过 img-only，说明 EEG 携带 image 不含的注意力信号。
- `freeze_img=True` (cross-attn 默认)：图片嵌入冻结，不允许 fusion 偷偷 fine-tune 视觉特征；Δ 增益只能来自 EEG。
- 训练评估都 `eeg_split=='test'` ── 沿用 session JSON 预切的划分。

---

## 文献复现 (knowledges/)

已实现：
- **EEGNet** ── Lawhence 2018, EEGNet 论文标准结构
- **ATM Encoder** ── Li et al. 2024 (`knowledges/li2024_atm_neurips.pdf`)，三阶段架构
- **Cross-attention fusion** ── 类似 ATM-S "EEG attends to image" 思路

参考但未完成实现：
- **NICE / NICE-EEG** (`knowledges/eeg_foundation_challenge_neurips2025.pdf`) ── self-supervised contrastive
- **ENIGMA** (`knowledges/enigma_neurips2025.pdf`, `enigma_openreview.pdf`) ── foundation model for EEG
- **NEED** (`knowledges/need_neurips2025.pdf`) ── attention-modulated EEG decoding
- **EEG2Video** (`knowledges/eeg2video_neurips2024.pdf`) ── EEG → diffusion video

---

## 当前阶段成果汇总

✅ 已完成：
1. 数据 IO 完成 join (6376/6376 epochs 100% 拿到 HINT)
2. 6 类 QC 图 + 多标签共现 + temporal sweep
3. EEGNet / ATM / Fusion 三模型 + 三 baseline 训练框架
4. 全部代码 `py_compile` 通过

⚠️ 沙箱内未能跑完整 pipeline 的部分（需要在用户机器上运行）：
- 加载完整 epochs (.fif=1.3GB) 需要 >4GB RAM
- torch / CLIP 嵌入需要 GPU
- 数值结果待用户机器输出

---

## 未完成 / 下一步推荐

### 高优先级（立刻做）
1. **在用户机器上运行 `process1.run_qc`** ── 拿到 ERP 实际形态、QC 分布、temporal sweep 峰值时间窗。如果 sweep 峰值不在 [100, 400] ms 区间，说明时间对齐/打标可能有问题。
2. **跑 `process2.img_embeddings`** ── 445 张唯一图片走 CLIP，约 30s on GPU。
3. **跑 `process2.compare_baselines`** ── 30 epochs，预计 GPU 15 min。看三 Δ：
   - Δ(eeg, img) > 0：EEG 携带独立信号
   - Δ(fusion, img) > Δ(eeg, img)：图片嵌入仍补充信息（非冗余）
   - Δ(fusion, eeg) > 0：fusion 不是 EEG-only 的简单上限

### 中优先级（结果出来再做）
4. **多标签实验**：每张图带 K 个 targets，用 binary cross-entropy 改 ImgOnly → 评估"图能解释多少 attention 选择"。
5. **Self-supervised pretrain**：把 `old_analysis/encoder/contrastive_wrapper.py` + Circle Loss 接到 ATM 上做 unsupervised pretrain（NICE 思路），再 fine-tune HINT 分类。
6. **跨被试 generalization**：等多个 subject 数据后，加 subject-aware batch norm 或 leave-one-subject-out 评估。
7. **时间分解**：把 fusion 的 cross-attention 权重按时间窗导出 ── 看 EEG 在哪个时间窗对 attention 贡献最大（应在 200~400ms P3a/P3b 区间）。

### 低优先级（待 Nature Comm 投稿前）
8. 单纯 image-only baseline 的 ceiling：用 LLM (CLIP text encoder) 加上 "Looking at X" prompt 评估纯视觉先验的上界。
9. ERP→eye-gaze 因果检验：fusion 中加 gaze_x/gaze_y 通道 vs 只用 EEG ── 看 fusion 增益是否来自眼动信号 leakage。
10. ENIGMA-style 大规模 foundation model 微调 ── 等有数据集 (>10 subjects) 时做。

### 已知潜在 pitfall
- **眼动 leakage**：epochs 含 `gaze_x`, `gaze_y` 通道。当前所有训练默认 `pick_eeg=True` 去掉它们；如果不小心带上，可能 30%+ 准确率来自眼动 ── 需在最终消融实验里显示「带 gaze vs 去 gaze」差异。
- **图片信息泄露**：每个 HINT 类别对应的 image set 可能在视觉空间上区分度很大（例如 "dog" 图都长得像狗），此时 image-only baseline 已经能解 >60%。这是 fair 的（实验设计如此），但需在论文中注明 image-only top-1 ~= 视觉先验 lower bound。
- **训练/测试集 image 是否独立**：当前 split 是 epoch-level (按 session JSON 的 `eeg_split`)，如果同一张图 ID 同时出现在 train 和 test 中，存在 image leakage。下一步要加 image-disjoint split 选项。

---

## 引用约定 (fact-check)
- EEGNet → Lawhence, V. J., Solon, A. J., et al. (2018). *J. Neural Eng.* 15:056013
- ATM    → Li, D., et al. (2024). *NeurIPS 2024.* arXiv:2403.07721
- NICE   → Song, Y., et al. (2023). *bioRxiv* (sustained 2024)
- 多标签共现矩阵的诊断意义见 `old_analysis/rsvp_quality_analysis.py` `extract_*` 系列
