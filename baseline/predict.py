from dataset import BottleSeg, A_VAL, get_model
import torch, cv2, pathlib, numpy as np

ROOT = "../data"
DEVICE= "cuda"


# for modality in ["rgb","pol","dif"]:

for modality in ["rgb", "dif", "pol"]:
    test_ds = BottleSeg(ROOT, "test", modality, A_VAL)
    # for kind in ["unet","deeplab"]:
    for kind in ["pspnet", "unet", "deeplab"]:
        model = get_model(kind).to(DEVICE)
        model.load_state_dict(torch.load(f"run/{kind}/{modality}/best.pt"))
        print(f"run/{kind}/{modality}/best.pt")
        model.eval()

        outdir = pathlib.Path(f"preds/{kind}/{modality}/")
        outdir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for img, mask, name in test_ds:

                pred = (model(img.unsqueeze(0).to(DEVICE)) > 0).squeeze().cpu().numpy().astype("uint8") * 255
                resized = cv2.resize(pred, (1224, 1024), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(outdir / f"{name}.png"), resized)
                print(f"Predicted {name}...")
