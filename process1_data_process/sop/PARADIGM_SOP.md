# 范式改进 SOP — Attention-RSVP-LVIS v2

**目的**: 在 N≥8 被试正式数据采集前，把 zfn-0507 pilot 暴露的 5 个数据质量问题（GFP peak 偏早、alpha 反向、SOA overlap、gaze artefact、RMS bimodality）全部解决。
**适用人员**: 数据采集操作员 + 范式工程师 + 数据 QC 分析师。
**版本**: v1.0, 2026-05-13

---

## 0. 总目标（pass/fail 阈值）

每次正式采集结束 24 h 内运行 `run_qc.py --config configs/process1_clean.yaml` 自动跑完，**所有数字必须达到下表才算合格**：

| 指标 | Pass | Soft-fail | Hard-fail |
|---|---|---|---|
| GFP peak latency | 80-130 ms | 130-160 ms | < 80 或 > 160 |
| Alpha SNR post/pre | 0.55-0.95 | 0.95-1.10 | > 1.10 |
| GFP_pre / GFP_post ratio | < 0.7 | 0.7-0.9 | > 0.9 |
| RMS bimodality coefficient | < 0.40 | 0.40-0.55 | > 0.55 |
| Saccade-contaminated trial rate (>2° in [0,300]ms) | < 5% | 5-15% | > 15% |
| Bad-trial rate (kurt/p2p) | < 3% | 3-8% | > 8% |
| Class balance min/max ratio | > 0.7 | 0.5-0.7 | < 0.5 |

**Hard-fail ⇒ 当次采集废弃，本被试当天不重测**（疲劳影响下一 session）。
**Soft-fail ⇒ 当次保留但标记 `quality_flag: soft`，纳入 sensitivity analysis**。

---

## 1. 采集前一次性校准（每台机/每月一次）

### 1.1 Photodiode-marker latency 测量（修问题 #2）

**为什么**：当前 GFP 峰在 52 ms，远早于生理 P1。怀疑显示器 → 触发 marker 之间有未补偿延迟。

**步骤**：
1. 准备 photodiode 探头（吸盘式），固定在显示器左上角刺激覆盖区
2. 运行 `experiment/calibration/photodiode_latency.py`（**TODO: 待实现**），在屏幕同一位置呈现 200 次黑↔白翻转，每次发 TTL marker
3. 同步采集 photodiode 信号到 `adc_*.bin`（已有的 ADC 通道接 photodiode 即可）
4. 解析延迟：`Δt = t_photodiode_rise − t_marker`
5. 报告 `mean ± std`，并写入 `process1_data_process/configs/base.yaml`:
   ```yaml
   acquisition:
     photodiode_delay_ms: 27.3   # 实测，每台机不同
   ```
6. 在 `data_io.load_epochs()` 中对 `times` 加上 `+photodiode_delay_ms / 1000`

**Pass 条件**：std/mean < 0.10（抖动 < 10%）；mean 在 15-60 ms 之间

### 1.2 显示器刷新率与刺激持续时间对齐

**为什么**：当前 refresh_hz=360 Hz，stim duration 350 ms ≈ 126 帧。如果显示器实际只有 60/144 Hz，会有 tearing。

**步骤**：
1. 用 photodiode 测量实际刷新周期 `Δt_refresh`
2. 验证 `stim_duration / Δt_refresh` 为整数（误差 < 1 ms）
3. 在 `paradigm.yaml` 中写入实测值

### 1.3 电极位置标定（修问题 #4）

**为什么**：当前 4×8 patch 显示 alpha post/pre = 1.44（反向），怀疑 patch 没覆盖 O1/Oz/O2 关键 alpha 生成区。

**新增协议 — Alpha-localizer block**（每被试上电极后**先做 5 分钟**）：
1. 被试睁眼凝视黑屏中心 60 s ⇒ `eyes_open_baseline`
2. 闭眼 60 s ⇒ `eyes_closed_baseline`  
3. 重复 3 次
4. 实时计算每电极 closed/open alpha-power ratio
5. **Pass 标准**：至少 4 个后枕电极 ratio > 2.5（典型 alpha modulation）
6. **Fail 处理**：把 patch 向后下方移动 1-2 cm，重测；若仍 fail，本被试今天放弃

`experiment/calibration/alpha_localizer.py`（**TODO: 待实现**）

---

## 2. 每被试每次采集（在椅子上）

### 2.1 上电极 → 阻抗 → Alpha-localizer

| 步骤 | 时长 | Pass 标准 | Fail 处理 |
|---|---|---|---|
| 量阻抗 | 2 min | 所有 ch < 20 kΩ | 涂 paste，重测；3 次不过则换 patch |
| Alpha-localizer (1.3) | 5 min | 后枕 ratio > 2.5 | 调整 patch 位置 |
| Gaze calibration | 2 min | 9-point 平均误差 < 1° | 重做；> 2 次则放弃眼动数据 |
| 范式说明 + 练习 block | 3 min | 被试正确率 > 80% (sample 30 trials) | 多练 1-2 个 block |

### 2.2 范式时序参数（修问题 #3 — 加 SOA jitter）

**当前**：固定 SOA = 514 ms ⇒ rERP unidentifiable from continuous EEG

**改为**：
```yaml
paradigm:
  stim_duration_ms: 350           # 保持
  isi_ms_min: 100                 # 改！下限
  isi_ms_max: 300                 # 改！上限
  isi_ms_distribution: "uniform"  # 改！
  # 实际 SOA = 350 + uniform(100, 300) = uniform(450, 650) ms
```

**理由**：均匀 200 ms jitter 让相邻 trial 不再周期，rERP 设计矩阵 A 变为 full-rank → **不再依赖 finite-support prior**，反卷积质量提升。Smith & Kutas 2015 推荐 jitter ≥ 0.5×SOA。

### 2.3 ITI / break 协议（修问题 #5 — RMS bimodality）

**当前**：30+ min 连续记录可能导致 patch 接触漂移

**改为**：
- 每 5 个 RSVP block (≈ 5 min) 插入 30 s 固定休息（不许动）
- 每 15 min 插入 2 min 主动休息（可以闭眼、不能说话）
- 记录员每 10 min 巡视一次电极/导线位置
- 在 session JSON 增加 `block_rest_*` events（如已有可复用）

### 2.4 注视约束（修问题 #6 — gaze artefact）

**当前**：眼动无 artefact rejection

**改为**：
1. 屏幕中心始终有红色 fixation cross（0.5° × 0.5°），整个 trial 期间显示
2. 每个 trial 实时检测 gaze：若 [0, +300] ms 内 gaze 偏离中心 > 2°，trial 被 mark 为 `saccade_contam=True`（写入 session JSON）
3. 后处理 `process1_data_process` 默认 reject 这些 trial
4. 每 block 结束反馈给被试："请保持注视中心"
5. **Pass 标准**：saccade_contam rate < 5%（per block）

`experiment/eyetracking/online_fixation_monitor.py`（**TODO: 待实现** — 当前已有 eye_csv 后处理但无 online）

---

## 3. 采集后自动 QC（24h 内）

### 3.1 自动管线

```bash
# 1. 跑主管线
python -B -m process1_data_process.run_qc \
  --config configs/process1_clean.yaml > logs/run_qc_${SUBJ}.log 2>&1

# 2. 跑 overlap 校正
python -B -m process1_data_process.run_overlap \
  --config configs/process1_overlap.yaml >> logs/run_overlap_${SUBJ}.log 2>&1

# 3. 跑 QC gate 检查（TODO: 待实现）
python -B -m process1_data_process.qc_gate \
  --summary process1_data_process/fig_clean/process1_summary.json \
  --thresholds process1_data_process/configs/qc_gates.yaml \
  --out process1_data_process/fig_clean/QC_REPORT.md
```

### 3.2 QC gate 评估清单

| 指标 | 来源 | 阈值 |
|---|---|---|
| `n_bad_trials / n_epochs` | `qc_summary` | < 3% pass / < 8% soft / ≥ 8% hard-fail |
| `grand_average_stats.gfp_peak_ms` | `process1_summary.json` | 80-130 ms pass / 130-160 soft / else hard |
| `alpha_snr_post_pre_mean` | 同上 | < 0.95 pass / < 1.10 soft / else hard |
| `attention_stats.diff_peak_ms` | 同上 | 100-300 ms pass |
| `attention_stats.frac_sig_post` | 同上 | > 0.10 pass |
| RMS bimodality (Hartigan dip test) | 重算 | < 0.40 pass / < 0.55 soft |
| Saccade-contam rate | session JSON | < 5% pass / < 15% soft |
| Class balance min/max | `class_balance.hint` | > 0.7 pass |

### 3.3 决策树

```
                    QC gates evaluated
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           all PASS    1-2 SOFT    any HARD-FAIL
              │           │             │
              ▼           ▼             ▼
    ✓ enrol into     ⚠ flag in        ✗ discard
      paper N        sensitivity        re-record next
                     analysis           visit
                     (sub-N table)      (≥ 48h apart)
```

---

## 4. 数据上传与版本化

每次采集合格后：

```bash
# 1. 数据文件夹结构
data/<subj_id>-<YYYYMMDD>/
├── adc_*.bin                       # raw photodiode + TTL
├── adc_*.meta                      # ADC config
├── data-*.npy                      # raw EEG (mne convention)
├── epochs_big-epo.fif              # MNE epoched
├── session_*.json                  # paradigm events + per-trial metadata
├── eye_*.csv                       # eyetracker raw
├── alpha_localizer_*.json          # NEW: pre-recording alpha test
└── photodiode_calib_*.json         # NEW: latency measurement

# 2. 在 process1 配置里改 dataset paths
configs/process1_clean.yaml:
  dataset:
    fif:          "data/<subj_id>-<YYYYMMDD>/epochs_big-epo.fif"
    session_json: "data/<subj_id>-<YYYYMMDD>/session_*.json"

# 3. 跑全套
bash scripts/run_full_subject_pipeline.sh <subj_id> <YYYYMMDD>

# 4. 提交结果到 Box / Notion
- fig_clean/ 全部 PNG
- fig_overlap/ 全部 PNG
- process1_summary.json
- QC_REPORT.md
```

---

## 5. 多被试 batch SOP

### 5.1 招募约束
- 年龄 18-35（avoid α-band age effects）
- 正常 / 矫正视力 > 1.0
- 无神经精神病史
- 当天睡眠 ≥ 6 h（drowsy 干扰 alpha）
- 录制时间窗：上午 10-12 或 下午 14-16（避开餐后困倦）

### 5.2 排除标准（前置）
- 头围太小 / 太大无法贴合 4×8 patch
- 头皮过油 / 头发太厚导致阻抗 > 30 kΩ 不可降
- Alpha-localizer 三次调整仍 fail

### 5.3 进度跟踪
```
N=1: zfn-0507  ✓ (pilot, 部分指标 fail，作为 method-fig 用例)
N=2: TBD       (需先实现 1.1 / 1.3 / 2.2 / 2.4 的改进)
...
N=8: 论文最少样本
```

---

## 6. 三个改进的代码 TODO（必须在 N=2 前完成）

| 优先级 | 模块 | 文件 |
|---|---|---|
| P0 | photodiode latency 校准脚本 | `experiment/calibration/photodiode_latency.py` |
| P0 | alpha-localizer 实时反馈 | `experiment/calibration/alpha_localizer.py` |
| P0 | jittered ISI in paradigm | `experiment/paradigms/rsvp_attention.py` 改 `isi_ms` 为 `uniform(100, 300)` |
| P1 | 在线 fixation monitor | `experiment/eyetracking/online_fixation_monitor.py` |
| P1 | gaze-based trial rejection | `process1_data_process/qc_signal.py` 加 `gaze_qc()` |
| P1 | QC gate 自动评估器 | `process1_data_process/qc_gate.py` |
| P2 | 多被试聚合 | `process1_data_process/aggregate_group.py` |

---

## 7. 期望改进幅度（采集 N≥8 后）

| 指标 | zfn-0507 (v1) | v2 期望（修完 1-4 后） |
|---|---|---|
| GFP peak latency | 52 ms | 95-110 ms |
| Alpha SNR post/pre | 1.44 | 0.6-0.8 |
| GFP_pre / post ratio | 1.01 | 0.4-0.6 |
| Salience Δ peak (rERP) | -0.5 µV | -1.5 µV @ 170 ms |
| HINT top-1 LDA | 0.076 | 0.10-0.15 |
| HINT top-5 LDA | 0.276 | 0.35-0.45 |
| Saccade-contam rate | unknown | < 5% |
| rERP cond | 7.0 (with prior) | < 5 (no prior, jittered SOA) |

---

## 8. 复现与可追溯

- 每次 run 自动落 `_resolved_config.yaml` 到 `fig_*/` 目录
- session JSON 版本号写入 `paradigm_version: "v2.0"` 字段
- 数据 commit hash 写入 `process1_summary.json.code_version`
- 论文 supplementary 附本 SOP 全文 + 所有 yaml config + 代码 commit

