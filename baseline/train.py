import argparse, time, torch, pathlib
from torch.utils.data import DataLoader
from dataset import BottleSeg, A_TRAIN, A_VAL, get_model
from torchmetrics.classification import BinaryJaccardIndex
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def train(root, model_type, modality, epochs=15, bs=8, device="cuda:0"):
    train_ds = BottleSeg(root, "train", modality, A_TRAIN)
    val_ds   = BottleSeg(root, "val",   modality, A_VAL)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds, batch_size=bs, shuffle=False)

    outdir = pathlib.Path(f"../runs/{run_id}/{model_type}/{modality}")
    outdir.mkdir(parents=True, exist_ok=True)

    model = get_model(model_type).to(device)
    opt   = torch.optim.AdamW(model.parameters(), 1e-4)
    best  = 0.0

    writer = SummaryWriter(log_dir=str(outdir / "logs"))
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logit = model(x)
            loss  = F.binary_cross_entropy_with_logits(logit, y)
            loss.backward(); opt.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dl)
        writer.add_scalar("Loss/train", avg_train_loss, ep)

        metric = BinaryJaccardIndex().to(device)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_dl:
                x, y = x.to(device), y.to(device)
                logit = model(x)
                loss = F.binary_cross_entropy_with_logits(logit, y)
                val_loss += loss.item()
                p = (logit > 0).float()
                metric.update(p, y)
        miou = metric.compute().item()
        avg_val_loss = val_loss / len(val_dl)
        writer.add_scalar("mIoU/val", miou, ep)
        writer.add_scalar("Loss/val", avg_val_loss, ep)

        if miou > best:
            best = miou
            torch.save(model.state_dict(), outdir / "best.pt")
            print(f"Device {device} Epoch {ep+1}/{epochs}  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}  mIoU: {miou:.3f} <-- Best model saved!")
        else:
            print(f"Device {device} Epoch {ep+1}/{epochs}  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}  mIoU: {miou:.3f}")
        torch.save(model.state_dict(), outdir / "last.pt")
        writer.flush()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--model_type", required=True, choices=["pspnet", "unet", "deeplab"])
    parser.add_argument("--modality", required=True, choices=["rgb", "dif", "pol"])
    args = parser.parse_args()

    args = parser.parse_args()

    run_id = args.run_name or time.strftime("%Y%m%d-%H%M%S")

    train(args.root, args.model_type, args.modality, args.epochs, args.bs, args.device)
