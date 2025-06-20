
import torch, torch.nn.functional as F, pathlib
from torch.utils.data import DataLoader
from dataset import BottleSeg, A_TRAIN, A_VAL, get_model
from torchmetrics.classification import BinaryJaccardIndex
from torch.utils.tensorboard import SummaryWriter
import time

ROOT = "../data"
EPOCHS = 15
BS     = 8
DEVICE = "cuda:1"


for kind in ["pspnet", "unet", "deeplab"]:
    for modality in ["rgb", "dif", "pol"]:
        # data
        start_time = time.time()
        print(f"{start_time} - Starting loading for {kind} on {modality} modality...")
        train_ds = BottleSeg(ROOT, "train", modality, A_TRAIN)
        val_ds   = BottleSeg(ROOT, "val",   modality, A_VAL)
        train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4)
        val_dl   = DataLoader(val_ds, batch_size=BS, shuffle=False)

        # model & optim
        model = get_model(kind).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), 1e-4)
        best  = 0.0

        logdir = f"run/{kind}/{modality}/tb"
        writer = SummaryWriter(log_dir=logdir)
        # Print time

        train_time = time.time()
        print(f"{train_time} - Starting training for {kind} on {modality} modality...")
        print(f"Loading took {train_time - start_time:.2f} seconds")

        for ep in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for x, y, _ in train_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logit = model(x)
                loss  = F.binary_cross_entropy_with_logits(logit, y)
                loss  = F.binary_cross_entropy_with_logits(logit, y)
                loss.backward(); opt.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_dl)
            writer.add_scalar("Loss/train", avg_train_loss, ep)

            # quick val IoU
            metric = BinaryJaccardIndex().to(DEVICE)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y, _ in val_dl:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logit = model(x)
                    loss = F.binary_cross_entropy_with_logits(logit, y)
                    val_loss += loss.item()
                    p = (logit > 0).float()
                    metric.update(p, y)
            miou = metric.compute().item()
            avg_val_loss = val_loss / len(val_dl)
            writer.add_scalar("mIoU/val", miou, ep)
            writer.add_scalar("Loss/val", avg_val_loss, ep)

            pathlib.Path(f"run/{kind}/{modality}").mkdir(parents=True, exist_ok=True)
            if miou > best:
                best = miou
                print(f"New best mIoU {miou:.3f}")
                # append best.pt to tthe path
                torch.save(model.state_dict(), f"run/{kind}/{modality}/best.pt")
            torch.save(model.state_dict(), f"run/{kind}/{modality}/last.pt")
            print(f"{kind} {modality}  epoch {ep:02d}  mIoU {miou:.3f}")
            writer.flush()

        writer.close()
