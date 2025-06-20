import segmentation_models_pytorch as smp

from torch.utils.data import Dataset
import cv2, numpy as np, torch, pathlib
import albumentations as A

class BottleSeg(Dataset):
    def __init__(self, root, split, modality="rgb", aug=None):
        self.img_dir  = pathlib.Path(root) / modality / "images" / split
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")
        self.mask_dir = pathlib.Path(root) / "masks" / split
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory {self.mask_dir} does not exist.")

        self.ids = sorted(p.stem for p in self.img_dir.glob("*"))
        self.aug = aug or A.Compose([], additional_targets={"mask": "mask"})

    def __len__(self):  return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img  = cv2.imread(str(self.img_dir / f"{name}.png"))[..., ::-1]  # BGR→RGB
        msk  = cv2.imread(str(self.mask_dir / f"{name}_mask.png"), cv2.IMREAD_GRAYSCALE)
        msk  = (msk > 127).astype("float32")  # 0/1

        transformed = self.aug(image=img, mask=msk)
        img = transformed["image"].astype("float32") / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)          # C,H,W
        msk = torch.from_numpy(transformed["mask"]).unsqueeze(0)  # 1,H,W
        return img, msk, name


A_TRAIN = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Resize(height=1024, width=1024) # deterministic resizing
], additional_targets={"mask":"mask"})

A_VAL   = A.Compose([A.Resize(height=1024, width=1024)],
                    additional_targets={"mask":"mask"})


def get_model(kind):
    if kind == "unet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=1,
            activation=None,
        )

    if kind == "deeplab":
        return smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=1,
            activation=None,
        )

    if kind == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=1,
            activation=None,
        )

    if kind == "pspnet":
        return smp.PSPNet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=1,
            activation=None,
        )

    if kind == "segformer":
        # lightweight transformer implementation via HuggingFace
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        cfg = SegformerConfig(
            num_labels=1,                # binary mask → 1 logit channel
            id2label={0: "foreground"},
            label2id={"foreground": 0},
        )
        return SegformerForSemanticSegmentation(cfg)

    raise ValueError(f"Unknown model kind: {kind}")