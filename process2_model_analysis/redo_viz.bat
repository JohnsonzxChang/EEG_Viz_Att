@echo off
cd /d C:\Users\thlab\Documents\Claude\Projects\EEG_Viz_Att
C:\Users\thlab\.conda\envs\VIZ\python.exe -u -m process2_model_analysis.redo_viz ^
  --subjects zfn-0507 zxc-0516 ^
  --root . ^
  --selection experiment/stimuli_select/stimuli_rsvp_attention_lvis_pilot_20260506_164009.json ^
  --coco_root C:/Users/thlab/Desktop/ES_coco/data/coco ^
  --fig_dir process2_model_analysis/fig ^
  --top_k 30
