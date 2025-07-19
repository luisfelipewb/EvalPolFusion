import os
import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from semseg.models import *
from semseg.utils.utils import timer
import numpy as np
from semseg.datasets.potato_multi_modality import POTATOMM
import time
class SemSeg:
    def __init__(self, cfg):
        self.device = torch.device(cfg['DEVICE'])
        dataset_cls = eval(cfg['DATASET']['NAME'])
        self.palette = dataset_cls.PALETTE

        modals = cfg['DATASET']['MODALS']
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette), modals)

        msg = self.model.load_state_dict(torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu'))
        print(msg)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    @timer
    def model_forward(self, input_list):
        # input_list: [Tensor(B,3,H,W), Tensor(B,3,H,W), ...]
        return self.model(input_list)

    def postprocess(self, orig_rgb, seg_map, overlay):
        # seg = seg_map.softmax(dim=1).argmax(dim=1).cpu()
        # seg_img = self.palette[seg].squeeze()
        seg = seg_map.softmax(dim=1).argmax(dim=1).cpu()[0]  # (H, W)
        seg_img = self.palette[seg.numpy()]  # (H, W, 3)
        if overlay:
            orig_rgb = orig_rgb.permute(1, 2, 0).cpu().numpy()
            seg_img = (orig_rgb * 0.4 + seg_img * 0.6).dtype(np.uint8)
        else:
            seg_img = seg_img.numpy().astype(np.uint8)
        # H, W = seg_img.shape[:2]
        # full_w = W + 0
        # padded_img = np.zeros((H, full_w, 3), dtype=np.uint8)
        # padded_img[:, 0:] = seg_img

        # return Image.fromarray(padded_img)

        return Image.fromarray(seg_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--cfg', type=str, required=True)


    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    save_dir=Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    semseg = SemSeg(cfg)

    dataset = POTATOMM(
        root=cfg['DATASET']['ROOT'],
        split='test',
        modals=cfg['DATASET']['MODALS']
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=cfg.get('NUM_WORKERS', 4), pin_memory=True)

    # save_dir = Path(cfg['SAVE_DIR']) / 'preds'
    target_size = (1224, 512)
    total_time=0
    for idx, (modal_list, _,filename) in enumerate(dataloader):
        # modal_list is a list of B×3×H×W tensors
        modal_list = [t.to(semseg.device) for t in modal_list]
        # RGB is first modality
        rgb = modal_list[0][0].cpu()

        starting_time=time.time()
        seg_out = semseg.model_forward(modal_list)
        process_time=time.time()-starting_time

        total_time+=process_time
        seg_img = semseg.postprocess(rgb, seg_out, cfg['TEST'].get('OVERLAY', False))
        seg_img = seg_img.resize(target_size, resample=Image.NEAREST)

        fname = Path(filename[0]).stem + ".png"
        seg_img.save(save_dir / fname)
        if idx %10==0:
            print("Saved:", save_dir / fname)
    mean_time=total_time/len(dataloader)
    print("Mean inference time :",mean_time)
if __name__ == "__main__":
    main()

