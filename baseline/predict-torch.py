from dataset import BottleSeg, A_VAL, get_model
import torch, cv2, pathlib, numpy as np

ROOT = "../data"
DEVICE= "cuda"


# for modality in ["rgb","pol","dif"]:
for modality in ["rgb"]:
    test_ds = BottleSeg(ROOT, "train", modality, A_VAL)
    # for kind in ["unet","deeplab"]:
    for kind in ["deeplab"]:
        mdl = get_model(kind).to(DEVICE)
        mdl.load_state_dict(torch.load(f"run/deeplab/rgb/best.pt"))
        mdl.eval()

        outdir = pathlib.Path(f"preds/{kind}/{modality}/train")
        outdir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            # for img, _ in test_ds:
            #     name = test_ds.ids.pop(0)  # same order as dataset iteration
            #     pred = (mdl(img.unsqueeze(0).to(DEVICE)) > 0).squeeze().cpu().numpy().astype("uint8")*255
            #     cv2.imwrite(str(outdir/f"{name}.png"), pred)

            for idx, (img, _) in enumerate(test_ds):
                name = test_ds.ids[idx]
                pred = (mdl(img.unsqueeze(0).to(DEVICE)) > 0).squeeze().cpu().numpy().astype("uint8")*255
                cv2.imwrite(str(outdir/f"{name}.png"), pred)
