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

        # ImageNet normalization values
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self):  return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img = cv2.imread(str(self.img_dir / f"{name}.png"))[..., ::-1]  # BGR→RGB
        msk = cv2.imread(str(self.mask_dir / f"{name}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(f"Missing image or mask for {name}")

        msk = (msk > 127).astype("float32")  # 0/1

        transformed = self.aug(image=img, mask=msk)
        img = torch.from_numpy(transformed["image"]).permute(2,0,1).float() / 255.0
        img = (img - self.mean) / self.std # Normalize using ImageNet mean/std
        msk = torch.from_numpy(transformed["mask"]).unsqueeze(0).float()
        return img, msk, name


offset = 384
height = 512
resize_dim = 512

A_TRAIN = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Crop(x_min=0, y_min=offset, x_max=1224, y_max=offset+height),
    A.Resize(height=resize_dim, width=resize_dim)
], additional_targets={"mask": "mask"})

A_VAL   = A.Compose([A.Crop(x_min=0, y_min=offset, x_max=1224, y_max=offset+height),
                     A.Resize(height=resize_dim, width=resize_dim)],
                    additional_targets={"mask":"mask"})

