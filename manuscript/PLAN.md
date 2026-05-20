# Manuscript v0 → v1 → submission PLAN
*Created: 2026-05-20  ·  Owner: jc  ·  Target venue: Science Advances*

本 PLAN 是 `manuscript/main.tex` (v0 草稿) 之后的 next-steps 清单。
v0 已经把 zfn-0507 + zxc-0516 两个 pilot 的真实数字 (top-1 / top-5 /
ΔERP peak / GFP cond / rERP λ / n_trials / class balance) 全部嵌入正文。
v0 编译需要 `C:\Program Files\MiKTeX\miktex\bin\x64` 在 PATH 中,
然后在 `manuscript/` 目录运行 `build.bat`。

---

## 0. v0 已经完成的事项 ✓

- [x] Sci Adv LaTeX 模板 (scifile.tex 改写) + scicite.sty + Science.bst 已落地 `manuscript/sty/`
- [x] `main.tex` 全部章节填充: Abstract / Introduction / Results
      (5 paragraphs) / Discussion / Materials and Methods / 4 张主图
- [x] `refs.bib` 含 24 条核心 + baseline + confound + benchmark 文献
- [x] 真实数据嵌入: 6376 epochs, 40 classes (105–205 trials),
      target/non-target ΔERP $=-1.75\,\mu$V @ 172 ms,
      rERP cond $=7.0$, fusion top-1 = 30.67 / 29.63\%,
      img-only = 22.98 / 20.52\%, EEG-only = 4.94 / 4.36\%
- [x] 16 张关键 PNG 复制到 `manuscript/figures/`
- [x] `build.bat` MiKTeX 编译脚本

---

## 1. 立刻需要做的事 (在投稿前 0–2 周内,P0)

### 1.1 编译验证 v0
```bat
cd C:\Users\thlab\Documents\Claude\Projects\EEG_Viz_Att\manuscript
build.bat
```
预期产出 `build/main.pdf`。**第一次编译可能因为 BibTeX 找不到
`sty/Science.bst` 报错**, 解决方案二选一:
- 把 `sty/Science.bst` 复制到 `build/` (BibTeX 工作目录), 或
- 改 `main.tex` 末尾的 `\bibliographystyle{sty/Science}` 为
  `\bibliographystyle{Science}`, 并把 `Science.bst` 拷到 `build/`。

### 1.2 sanity-check 数字
打开 `process1_data_process/fig_clean/process1_summary.json` 与
`process2_model_analysis/fig/results_digest.md`, 对照 main.tex
里的每一个百分比/微伏数:
- [ ] `n_epochs = 6376` ✓
- [ ] `n_target = 3189`, `n_nontarget = 3187` ✓
- [ ] `diff_peak_uv = -1.75`, `diff_peak_ms = 172` ✓
- [ ] `sig_window = [152, 516]` ms ✓
- [ ] `class_balance` min = 105 (bottle), max = 205 (laptop) ✓
- [ ] `n_unique_images = 445` ✓
- [ ] rERP: `A_shape = [376, 138]`, `cond = 7.0`, `ridge = 1.0` ✓
- [ ] process2 top-1/top-5 数字两个被试都对齐 ✓

### 1.3 zxc-0516 process1 跑一遍
目前 process1 的 fig_clean/ 和 fig_overlap/ 只跑了 zfn-0507。
zxc-0516 也需要跑一遍以放到 supplementary:
```bat
python -B -m process1_data_process.run_qc ^
  --fif data\zxc-0516\epochs_big-epo.fif ^
  --session_json data\zxc-0516\session_*.json ^
  --fig_dir process1_data_process\fig_zxc_clean ^
  --crop_tmin -0.1 --crop_tmax 0.45
```

---

## 2. v0 → v1 主要补足 (在投稿前 2–6 周, P1)

### 2.1 数据采集 (核心 blocker)
**v0 只有 N=2,Sci Adv 期刊审稿人会立即要求 N≥8。** 按
`process1_data_process/sop/PARADIGM_SOP.md` v2.0 的 7 个 QC gates 跑:

| 被试编号 | 状态 | photodiode 校准 | alpha-localizer | jittered ISI | online fixation | QC gate |
|---|---|---|---|---|---|---|
| zfn-0507 | ✓ pilot | ✗ | ✗ | ✗ (固定 514ms) | ✗ | F (3 soft / 1 hard) |
| zxc-0516 | ✓ pilot | ✗ | ✗ | ✗ | ✗ | (未评估,做!) |
| N=3 | TBD | **MUST 做完** | **MUST** | **MUST** | **MUST** | pass |
| N=4–8 | TBD | 校准好 | localizer pass | jitter 100–300ms | online QC | pass |

P0 工程任务 (按 SOP TODO 表):
- [ ] `experiment/calibration/photodiode_latency.py` — 实测显示器延迟
- [ ] `experiment/calibration/alpha_localizer.py` — 上电极前 5 min 闭/睁眼验电极位置
- [ ] `experiment/paradigms/rsvp_attention.py` 把 `isi_ms` 改为 `uniform(100,300)`
- [ ] `experiment/eyetracking/online_fixation_monitor.py` — 实时拒绝 saccade trial

### 2.2 v1 必须新增的 results 段落
- **§2.7 Cross-subject generalisation (zero-shot)**:
  Leave-one-subject-out 训练 fusion model,报告 8 个被试每人作为 test
  set 的 top-1/top-5。预计:naive LOSO 5–10% top-1,加 subject-aware
  batch norm 或 NEED-style alignment 后 12–18%。

- **§2.8 Saliency-orthogonality control**:
  把 fusion 的 attention map 与 DeepGaze IIE saliency 做 Pearson
  相关,目标 r < 0.3 (证明不是 saliency 模型的退化)。Reviewer 必问。

- **§2.9 SDiv (Subjective Divergence) metric**:
  $\text{SDiv} = \frac{1}{N_{\text{img}}}\sum_{\text{img}}\frac{1}{|C|^2}\sum_{c_i,c_j}\mathbb{1}[\hat{w}(c_i)\neq\hat{w}(c_j)]$
  SDiv=1 表示完全 cue-driven,SDiv=0 表示模型靠 image 推理。**这是
  范式合法性的硬核指标,必须报告。**

- **§2.10 EEG-ablation**:
  把 EEG 置零 / 替换为高斯噪声,看 SDiv → 0。如果 ablation 后
  fusion 准确率回落到 image-only,证明 EEG 不是装饰。

### 2.3 v1 必须新增的 controls (送审前 reviewer 一定会问)

| Reviewer 会问 | 我们的回应 | 状态 |
|---|---|---|
| "image content 是不是真的能解释 fusion 增益?" | EEG-ablation (§2.10) | ⏳ |
| "DeepGaze 就能输出同样的 attention map 吧?" | Saliency-orthogonality r < 0.3 (§2.8) | ⏳ |
| "你们 N=2 还敢投 Sci Adv?" | 完整 N≥8 + LOSO (§2.7) | **🚫 BLOCKER** |
| "Yao 2025 已经做过 attention 解码,你们 incremental?" | object-level vs video-level + 40-way vs 2-way 已在 Intro 强调,补一个 head-to-head 数据对比 | ⏳ |
| "gaze artefact 怎么排除的?" | online fixation monitor + post-hoc 拒绝 saccade trial + 把 gaze 作为协变量 | 部分 ⏳ |
| "GFP peak 52ms 怎么可能?" | photodiode 实测 delay 后修正 + 在 Methods 段已声明 caveat | ⏳ |
| "alpha SNR post/pre = 1.44?" | 在 Discussion 已声明为电极位置 artefact,full release 用 localizer 验证 | ✓ 已写 |
| "你们的 framing 不是 saliency 而是 selection,prove it!" | hint-swap + SDiv + EEG-ablation 三联证据 | 1/3 ✓ |

### 2.4 v1 数据 / 模型 ablation 表
扩 Results 现有 3 行表为完整 ablation matrix:

| Model              | Linear | EEGNet | ATM   | ATM+CLIP | Fusion (ours) |
|--------------------|--------|--------|-------|----------|---------------|
| Image-only baseline|        |        |       |          |               |
| EEG-only           | 7.6 (LDA) | TBD | 4.94  | -        | -             |
| EEG + Image        | -      |        |       |          | **30.67**     |
| EEG-zeroed + Image | -      |        |       |          | TBD           |
| EEG + Image-shuffled| -     |        |       |          | TBD           |

5×5 矩阵 + 跨被试 mean ± std + 95% CI。

---

## 3. v1 → submission 完成度 (P2)

### 3.1 Figure regeneration with proper styling
所有 figure 用统一字体 (Arial 8pt for axis, 9pt for labels)、
统一颜色 (process1/process2 plot_style.py 已经有 PALETTE 字典)、
统一 PNG/PDF (Science Adv 要求 figures 分别上传 high-res TIFF/EPS)。
- [ ] Fig 1: 重画 paradigm schematic (现在只有 dataset stats),需手画
      RSVP 时序图。Illustrator / matplotlib 都行。
- [ ] Fig 4B: cross-attention overlay,每个 subject 选 4 个 trial,
      展示 4 个 query 各自落在哪个 patch 上。**目前已有 zfn-0507 的 F4,
      需为 zxc-0516 也做一张**, 并在 caption 中说明 query→patch
      alignment 的 quantitative metric (Pearson r with object mask)。

### 3.2 Supplementary materials
Science Adv 允许 unlimited supplementary,建议至少:
- **SF1**: 40 categories 详细 ERP per-category atlas (`F05_erp_per_hint.png`)
- **SF2**: train/test ERP drift check (`F08_erp_split_drift.png`)
- **SF3**: QC distributions (`F09_qc_distributions.png`)
- **SF4**: Topomap snapshots at 50/100/170/250/350 ms (`F07_topomap_snapshots.png`)
- **SF5**: Alpha desync per HINT (`F12_alpha_desync.png`)
- **SF6**: temporal sweep LDA (`F17_temporal_sweep_hint.png`)
- **SF7**: rERP butterfly (`F19_rerp_butterfly.png`)
- **SF8**: rERP salient vs non-salient (`F20_rerp_salient_vs_nonsalient.png`)
- **SF9**: Process2 training curves (`F2_training_curves.png`)
- **SF10**: Confusion matrices per subject
- **Datasheet**: stimulus selection criteria,每张图的 (image_id,
  category, area, bbox) 表格,以及 LVIS class subset 文档。

### 3.3 Pre-registration
- [ ] OSF Stage 1 (paradigm + hypotheses): 应在 N≥3 之前公开
- [ ] OSF Stage 2 (analysis pipeline): 在 N≥3 之后、N=8 数据采集前公开
- [ ] 把两份 pre-reg link 写入 Methods

### 3.4 Ethics statement
- [ ] IRB approval number 填入 Methods
- [ ] 写一段 Data Availability statement (Zenodo / OSF DOI)
- [ ] Code repository: GitHub release + Zenodo DOI

---

## 4. 投稿策略与时间表

### 4.1 主投 / 备投
- **首选 (Tier-1)**: Science Advances (rolling, 6–8 周 first decision)
- **备选 (Tier-1.5)**: Nature Communications (rolling, 8–12 周)
- **备选 (Tier-2)**: PNAS, Nat Hum Behav, IEEE TBME
- **NeurIPS 2026 D&B Track** 可以并行投 dataset paper,
  与本主稿引用同一数据但 narrative 不同 (datasheet + benchmark 重点)。

### 4.2 时间表 (建议)
| 时间       | 事件                                                  |
|------------|-------------------------------------------------------|
| 2026-05-20 | v0 草稿完成 (本文档创建)                              |
| 2026-05-21 | v0 编译通过,内部 review                              |
| 2026-05–06 | N=3 至 N=4 数据采集 (含 paradigm v2 SOP 改进)         |
| 2026-06    | OSF Stage 1 pre-reg (在 N=4 之前)                     |
| 2026-06–08 | N=5 至 N=8 完成,zxc-0516 process1 重跑               |
| 2026-08    | v1 草稿: 加 LOSO + SDiv + EEG-ablation + Saliency 对照 |
| 2026-09    | 内部 fact-check + reviewer pressure-test              |
| 2026-09 末 | Science Advances 投稿                                 |
| 2026-12    | Reviewer 1st round 反馈,3 个月内出 v2                |
| 2027-03    | Accept (期望) / Re-review / Reject 备选 Nat Commun     |

### 4.3 与 Bertrand 组 (Yao 2025) 邮件协调
按 `paper_disclosure_note.md` §8 模板,在投稿前 4–6 周发邮件,
邀请 cross-paradigm validation 或 citation coordination。

---

## 5. 技术 debt / 已知 risk (按严重度排序)

| 风险 | 严重度 | 缓解 |
|---|---|---|
| **N=2 pilot 无法支持 Sci Adv 投稿** | 🔴 致命 | 完成 N≥8 (P0)|
| Photodiode delay 未校准 (GFP peak 52ms) | 🟠 中 | photodiode_latency.py 上线 |
| Alpha SNR post/pre = 1.44 (电极位置) | 🟠 中 | alpha-localizer 验证后再录 |
| RSVP 固定 SOA → rERP unidentifiable | 🟠 中 | jittered ISI 100–300 ms |
| Gaze artefact 未做 online rejection | 🟡 中 | online_fixation_monitor.py |
| Image leakage between train/test | 🟡 低 | 加 image-disjoint split 选项 |
| Fusion 增益可能来自 gaze 通道 leak | 🟡 低 | 已 pick_eeg=True 排除,Supp 中显式 ablation |
| 40 类 LVIS 类别中某些 HINT 的图像 visually 相似 (e.g. dog/cat) → image-only 已经能解 → 留给 EEG 的空间小 | 🟡 低 | 已在 Discussion 谈到;Supp 加 per-class breakdown |
| Reviewer 用 NEED 2025 / ENIGMA 2025 LOSO benchmark "压" 我们 | 🟡 低 | v1 加 NEED-style LOSO,直接报告 |

---

## 6. fact-check 清单 (投稿前最后一周必须做)

按用户偏好 (preferences 中: 对于前沿的科学观点和工程上的操作指引,需要在最后进行事实核查):

- [ ] **EEGNet** 引用: Lawhern et al. 2018 *J Neural Eng* 15:056013 ✓
- [ ] **ATM** 引用: Li et al. 2024 NeurIPS, arXiv:2403.07721 ✓
- [ ] **Yao 2025** 实际页码、DOI、Zenodo dataset ID
- [ ] **rERP** 引用: Smith & Kutas 2015 *Psychophysiology* 52(2):157
- [ ] **LVIS** 引用: Gupta et al. 2019 CVPR ✓
- [ ] **CLIP / ViT** 引用: Radford 2021, Dosovitskiy 2021 ✓
- [ ] **P1 timing** 引用: Foxe & Simpson 2002 *Exp Brain Res* 142:139 ✓
- [ ] **\#EEGManyLabs** 引用: Constant et al. 2025 *Cortex*
- [ ] **NSD / THINGS-EEG2** 引用 (作为代际对比的 dataset)
- [ ] **Stokes & Spaak 2016** *Trends Cogn Sci* 20(7):483 ✓
- [ ] 所有 PubMed / Scholar 链接到 DOI 而不是 PMID
- [ ] 所有 arXiv 引用带版本号
- [ ] Science.bst 输出格式与 Sci Adv 期刊近期 5 篇 paper 抽检比对

---

## 7. 反例测试 (self-critique drill,从 paper_disclosure_note §App A)

写完每个版本后自己压力测试:

1. *"如果只用 image 不用 EEG,模型还能输出 attention map 吗?"*
   → 现状 Δ(fusion vs img) = +7–9 pp 提示不能,但需要 SDiv 量化。
2. *"如果只用 EEG 不用 image,模型还能输出 attention map 吗?"*
   → EEG-only top-1 = 4.94%,远低于 image-only,所以 image 是合理 prior。
3. *"两个 subject 同图同 HINT,EEG-decoded attention 一致吗?"*
   → 等 N=2 之后做 cross-subject 同图同 cue 相关度分析。
4. *"同 subject 两个 trial 同图同 cue,attention map 一致性多高?"*
   → 试-试间一致性是 reliability 上限,要在 Methods 表中报告。
5. *"vs 单纯 cue → object mapping 优势在哪?"*
   → cue → object 在范式内 perfect,但价值是从 EEG 读出
   *attention engagement* 强弱,而非 cue identity。

---

## 8. 完整 commit 清单 (在投稿前)

```
manuscript/
├── main.tex            ✓ v0 (this run)
├── refs.bib            ✓ v0
├── PLAN.md             ✓ this file
├── build.bat           ✓ v0
├── sty/
│   ├── scicite.sty     ✓
│   └── Science.bst     ✓
├── figures/            ✓ 16 PNG copied (v0)
│   └── (TODO: replace with 600 dpi TIFF for final)
└── build/              (auto-generated by build.bat)
    └── main.pdf
```

---

**Final note**: v0 已经把所有真实数字、所有现有图、所有方法学
caveats 全部嵌入 main.tex。下一步的 P0 BLOCKER 是 N≥8 数据采集。
没有这个,任何 Tier-1 期刊都会一审拒。**先把范式 v2 SOP 跑通,
再投 manuscript。**
