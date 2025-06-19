
import torch, torch.nn.functional as F, pathlib
from torch.utils.data import DataLoader
from dataset import BottleSeg, A_TRAIN, A_VAL, get_model
from torchmetrics.classification import BinaryJaccardIndex

ROOT = "../data"
EPOCHS = 100
BS     = 16
DEVICE = "cuda:1"


# for modality in [ "pol", "dif"]:
    # for kind in ["unet"]:
for modality in ["rgb", "pol", "dif"]:
    for kind in ["unet", "deeplab"]:
        # data
        train_ds = BottleSeg(ROOT, "train", modality, A_TRAIN)
        val_ds   = BottleSeg(ROOT, "val",   modality, A_VAL)
        train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4)
        val_dl   = DataLoader(val_ds, batch_size=BS, shuffle=False)

        # model & optim
        model = get_model(kind).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), 1e-4)
        best  = 0.0

        for ep in range(EPOCHS):
            model.train()
            for x, y in train_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logit = model(x)
                loss  = F.binary_cross_entropy_with_logits(logit, y)
                loss.backward(); opt.step()

            # quick val IoU
            metric = BinaryJaccardIndex().to(DEVICE)
            model.eval()
            with torch.no_grad():
                for x,y in val_dl:
                    p = (model(x.to(DEVICE)) > 0).float()
                    metric.update(p, y.to(DEVICE))
            miou = metric.compute().item()
            pathlib.Path(f"run/{kind}/{modality}").mkdir(parents=True, exist_ok=True)
            if miou > best:
                best = miou
                print(f"New best mIoU {miou:.3f}")
                # append best.pt to tthe path 
                torch.save(model.state_dict(), f"run/{kind}/{modality}/best.pt")
            torch.save(model.state_dict(), f"run/{kind}/{modality}/last.pt")
            print(f"{kind} {modality}  epoch {ep:02d}  mIoU {miou:.3f}")
