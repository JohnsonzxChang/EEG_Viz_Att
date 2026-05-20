import os
import numpy as np
import pandas as pd
import glob
import re
import mne 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import sys 
from conf import BaseConfig
import random
from pycocotools.coco import COCO
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
try:
    import contrast.clips as _clips
    _HFCLIP_IMPORT_ERR = None
except Exception as e:
    _clips = None  # type: ignore
    _HFCLIP_IMPORT_ERR = e


warnings.filterwarnings('ignore')

ROOT_PATH = os.environ.get('DATA_ROOT', os.getcwd())
UDF_VIZ_PATH = f'{ROOT_PATH}/data/udf_viz'

from .data_loader import ALL_CLA

class Dataset_UDF_VIZ_Pair_multiclass(Dataset):
    def __init__(self, args:BaseConfig, seeds:int=None):
        super().__init__()
        self.args = args
        if seeds is not None:
            np.random.seed(seeds)
        self.only_img_ids = bool(getattr(self.args, "only_img_ids", False))
        # assert flag in ['trn', 'val', 'tst'], f"Flag must be one of ['trn', 'val', 'tst'], got {flag}"
        assert os.path.exists(UDF_VIZ_PATH), f"Dataset path {UDF_VIZ_PATH} does not exist"
        # all_files = [os.path.join(UDF_VIZ_PATH, 'zx-1122', 'erp.fif'), os.path.join(UDF_VIZ_PATH, 'zfn-1125', 'erp.fif')]
        # file_balence = [20, 16]
        # all_files = [os.path.join(UDF_VIZ_PATH, 'zfn-erp-1128', 'erp.fif')]
        # file_balence = [36]
        all_files = [os.path.join(UDF_VIZ_PATH, 'zx-1122', 'erp.fif')]
        file_balence = [20]
        assert len(all_files) > 0, f"No .fif files found in {UDF_VIZ_PATH}"
        self.all_data = []
        self.all_regs = []
        self.all_img_ids = []
        self.mega_data = [] # record the uniqe data mark for validation 
        self.all_img_emb = None
        self.all_cap_emb = None
        self.all_subjects = []

        coco = COCO(os.path.join(ROOT_PATH, 'data', 'coco', 'annotations', 'instances_train2017.json'))
        self.coco_cap = COCO(os.path.join(ROOT_PATH, 'data', 'coco', 'annotations', 'captions_train2017.json'))
        mlb = MultiLabelBinarizer(classes=np.arange(len(ALL_CLA)))
        mlb.fit([[]])  # fit once with known classes; use transform() in loop

        # Offline cache preferred (mature workflow)
        self.clip_cache_path = getattr(self.args, "clip_cache_path", None)
        self.clip_cache_strict = bool(getattr(self.args, "clip_cache_strict", True))
        self.clip_model = None
        self._clip_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for i, file in enumerate(all_files):
            epochs = mne.read_epochs(file, preload=True, verbose=False) # trials x chns x time (chns=32 + 3)
            dt, lb, ig, ca = [], [], [], []
            cnt = 0

            # Ultra-fast path: only collect COCO image ids from event codes.
            if self.only_img_ids:
                # epochs.event_id maps "<class>/<...>/<img>.jpg" -> code
                code_to_name = {code: name for name, code in epochs.event_id.items()}
                event_codes = epochs.events[:, 2]
                img_ids = []
                for c in event_codes:
                    name = code_to_name.get(int(c), "")
                    try:
                        img_ids.append(int(name.split('/')[-1].split('.')[0]))
                    except Exception:
                        continue
                self.all_img_ids.append(np.asarray(img_ids, dtype=int))
                continue
            
            for tgt in ALL_CLA:

                ep = epochs[tgt]
                code_to_name = {code: name for name, code in ep.event_id.items()}

                event_codes = ep.events[:, 2]  # (n_epochs,)
                img_id = [int(code_to_name[c].split('/')[-1].split('.')[0]) for c in event_codes]
                # img_id = 126050  # 你的图片 id（int）
                # 1) 找到这张图片对应的所有标注 id
                all_lblb = []
                all_img = []
                # all_cap = []
                for img in img_id:
                    ann_ids = coco.getAnnIds(imgIds=[img])
                    anns = coco.loadAnns(ann_ids)
                    # ann_ids_cap = coco_cap.getAnnIds(imgIds=[img])
                    # anns_cap = coco_cap.loadAnns(ann_ids_cap)

                    # 3) 从标注里提取 category_id，去重
                    cat_ids_in_img = sorted({ann["category_id"] for ann in anns})
                    # caps = [ann["caption"] for ann in anns_cap]
                    # print(cat_ids_in_img)

                    cats = coco.loadCats(cat_ids_in_img)
                    cat_names_in_img = [ALL_CLA.index(c["name"]) for c in cats if c["name"] in ALL_CLA]
                    all_lblb.append(cat_names_in_img)
                    all_img.append(img)
                    # all_cap.append(caps)
                    # print(cat_names_in_img)
                    # 例如: ['person', 'dog', 'car']
                all_lblb = mlb.transform(all_lblb)
                all_img = np.array(all_img, dtype=int)
                # all_cap = np.array(all_cap)
                
                tmp = ep.get_data()*1e6
                assert all_lblb.shape[0] == tmp.shape[0] == all_img.shape[0] and all_lblb.shape[1] == len(ALL_CLA) and len(all_lblb.shape) == 2, f'{all_lblb.shape}...{tmp.shape}...{all_img.shape}'
                if tmp.shape[0] >= file_balence[i]:
                    all_sel = np.random.choice(list(range(tmp.shape[0])), size=file_balence[i], replace=False)
                    dt.append(tmp[all_sel,:,:]) # # 89, 83, 17
                    lb.append(all_lblb[all_sel,:]) # 89, 83, 17
                    ig.append(all_img[all_sel])
                    # ca.append(all_cap[all_sel])
                else:
                    print(f'data class imbalance in class {tgt} with {tmp.shape[0]} shorter than {file_balence[i]}')
                    dt.append(tmp) # # 89, 83, 17
                    lb.append(all_lblb) # 89, 83, 17
                    ig.append(all_img)
                    # ca.append(all_cap)
                # dt.append(tmp)
                # lb.append(np.array([cnt]*tmp.shape[0]))
                cnt += 1

            # Fast path: only collect image ids for offline embedding precompute
            if self.only_img_ids:
                ig = np.concatenate(ig, axis=0)
                self.all_img_ids.append(ig.astype(int))
                continue
            dt = np.vstack(dt) # trials x chns x time
            lb = np.concatenate(lb, axis=0) # trials
            ig = np.concatenate(ig, axis=0) # trials
            # ca = np.concatenate(ca, axis=0) # trials
            print(dt.shape, lb.shape, ig.shape) # (420, 35, 1001) (420,)

            regVal = lb  # 0,1,2,3,4,5
            data = dt[:, self.args.chn_sel, :] # only use first chn_num channels 5501 = 1000 void + 4000 data + 501 void
            # regVal = epochs[:, 32+1:32+3, :] # y,z position

            # data = data[:,:,1200:3700]

            for j in range(1): # 4000 / 500 = 8 * 2 = 16 mux
                t_len = self.args.t_len
                # mid = 750 + 250 + j*(t_len//2)
                mid = self.args.t0 + t_len//2 + j*(t_len)
                start = mid - (t_len//2); end = mid + (t_len//2)
                print(f'start: {start}...end:{end}')
                self.all_data.append(data[:, :, start:end].astype(np.float32))
                self.all_regs.append(regVal.astype(np.float32)) # use the last time point as regression target
                self.all_subjects.append(np.ones((data.shape[0],), dtype=int) * i)
                self.all_img_ids.append(ig.astype(int))
                # self.all_caps.append(ca)
        self.all_img_ids = np.concatenate(self.all_img_ids, axis=0)

        if self.only_img_ids:
            # Minimal init for precompute workflow
            self.trn_sel = list(range(self.all_img_ids.shape[0]))
            self.val_sel = []
            self.flag = None
            return

        self.all_data = np.concatenate(self.all_data, axis=0)
        self.all_regs = np.concatenate(self.all_regs, axis=0)
        self.all_subjects = np.concatenate(self.all_subjects, axis=0)

        # Load offline CLIP cache (preferred)
        if self.clip_cache_path is None:
            clip_model_name = getattr(self.args, "clip_model_name", "openai/clip-vit-large-patch14")
            model_tag = clip_model_name.replace("/", "-").replace(" ", "")
            cand = os.path.join("./data/coco/processed_train2017", f"clip_embeds_{model_tag}.npz")
            if os.path.isfile(cand):
                self.clip_cache_path = cand

        if self.clip_cache_path is not None and os.path.isfile(self.clip_cache_path):
            print(f"Loading CLIP embeddings from cache: {self.clip_cache_path}")
            npz = np.load(self.clip_cache_path, allow_pickle=True)
            cache_ids = npz["img_ids"].astype(int)
            cache_img = npz["img_emb"].astype(np.float32)
            cache_cap = npz["cap_emb"].astype(np.float32)
            for i, iid in enumerate(cache_ids.tolist()):
                self._clip_cache[int(iid)] = (cache_img[i], cache_cap[i])

        unique_img_ids = sorted({int(x) for x in self.all_img_ids.tolist()})
        missing = [iid for iid in unique_img_ids if int(iid) not in self._clip_cache]
        if len(missing) > 0:
            if self.clip_cache_strict:
                raise FileNotFoundError(
                    f"Missing {len(missing)} image ids in clip cache. "
                    f"Set args.clip_cache_path to a precomputed npz or run scripts/precompute_coco_clip_embeds.py"
                )
            # Optional fallback to online compute
            if _clips is None:
                raise ImportError(
                    "contrast.clips is unavailable for online fallback. "
                    f"Original import error: {_HFCLIP_IMPORT_ERR}"
                )
            enc_name = "HF" + "CLIP" + "Encoder"
            if not hasattr(_clips, enc_name):
                raise ImportError("CLIP encoder unavailable; install transformers.")
            clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            clip_model_name = getattr(self.args, "clip_model_name", "openai/clip-vit-base-patch32")
            self.clip_model = getattr(_clips, enc_name)(model_name=clip_model_name, device=clip_device).to(clip_device)
            self.clip_model.eval()
            for iid in missing:
                img_emb, cap_emb = self._get_img_and_cap_emb(iid)
                self._clip_cache[int(iid)] = (img_emb, cap_emb)

        # Materialize per-sample embeddings
        self.all_img_emb = np.stack([self._clip_cache[int(iid)][0] for iid in self.all_img_ids], axis=0).astype(np.float32)
        self.all_cap_emb = np.stack([self._clip_cache[int(iid)][1] for iid in self.all_img_ids], axis=0).astype(np.float32)
        self.trn_sel = list(range(self.all_data.shape[0]))
        
        self.val_sel = np.random.choice(self.trn_sel, size=int(0.2*len(self.trn_sel)), replace=False)
        self.trn_sel = list(set(self.trn_sel) - set(self.val_sel))
        print(f"Total samples: {self.all_data.shape[0]}, Training samples: {len(self.trn_sel)}, Validation samples: {len(self.val_sel)}, top 4: {self.all_regs[:4,:]}")
        self.flag = None 

    def get_flag(self, flag:str):
        assert flag in ['trn', 'val', 'tst'], f"Flag must be one of ['trn', 'val', 'tst'], got {flag}"
        if flag == 'trn':
            self.flag = self.trn_sel
        elif flag == 'val':
            self.flag = self.val_sel
        else:
            raise ValueError(f"Flag must be 'trn' or 'val', got {flag}")
        print(f'Validating {self.val_sel[:10]} ... samples for {flag} set.')
        self.all_data = self.all_data[self.flag]
        self.all_regs = self.all_regs[self.flag]
        self.all_subjects = self.all_subjects[self.flag]
        self.all_img_ids = self.all_img_ids[self.flag]
        self.all_img_emb = self.all_img_emb[self.flag]
        self.all_cap_emb = self.all_cap_emb[self.flag]

    
    def __len__(self):
        return self.all_subjects.shape[0]
    
    def _get_img_and_cap_emb(self, img_id: int):
        img_root = getattr(self.args, 'coco_img_path', None)
        if img_root is None:
            if 'win' in sys.platform:
                img_root = None  # <SET YOUR DATA PATH>
            else:
                img_root = None  # set via config.coco_img_path
        img_id_int = int(img_id)
        file_id = str(img_id_int).zfill(12)
        path = os.path.join(img_root, file_id + ".jpg")
        assert os.path.isfile(path), f'{path} not exist ......'
        img = Image.open(path).convert("RGB")

        ann_ids = self.coco_cap.getAnnIds(imgIds=[img_id_int])
        anns = self.coco_cap.loadAnns(ann_ids)
        captions_all = [a["caption"] for a in anns if "caption" in a]
        caption = captions_all[0] if len(captions_all) > 0 else ""

        with torch.no_grad():
            out = self.clip_model(images=[img], texts=[caption])

        img_emb = out.image_embeds[0].detach().cpu().numpy()
        cap_emb = out.text_embeds[0].detach().cpu().numpy()
        return img_emb, cap_emb  # (D,), (D,)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.all_data[idx].copy())
        regs = torch.from_numpy(self.all_regs[idx])
        subjects = torch.tensor(int(self.all_subjects[idx]), dtype=torch.long)
        img_emb = torch.from_numpy(self.all_img_emb[idx])
        cap_emb = torch.from_numpy(self.all_cap_emb[idx])

        # EEG augmentation (training only) ─────────────────────────────────
        if self.flag is self.trn_sel:
            # 1. Gaussian noise (σ = 10% of signal std)
            data = data + torch.randn_like(data) * data.std() * 0.1
            # 2. Time masking: zero out a random 10% temporal segment
            T = data.shape[-1]
            mask_len = max(1, int(T * 0.1))
            t_start = random.randint(0, T - mask_len)
            data[:, t_start:t_start + mask_len] = 0.0
        # ────────────────────────────────────────────────────────────────────

        return {
            "data": data,
            "regs": regs,
            "subjects": subjects,
            "img_emb": img_emb,
            "cap_emb": cap_emb,
        }






def non_linear_transfer(x, kind='exp'):
    if kind == 'exp':
        return np.sign(x) * (np.exp(np.abs(x)) - 1)
    elif kind == 'log':
        return np.sign(x) * np.log(np.abs(x) + 1)