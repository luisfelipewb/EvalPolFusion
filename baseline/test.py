import argparse, pathlib, torch, cv2
from bottleseg.models import get_model
from bottleseg.data.bottle import BottleSeg, A_TRAIN, A_VAL
from bottleseg.models import registry
import time


def predict(root, run_id, modality, model_type, device="cuda"):
    weights_path = pathlib.Path(f"../runs/{run_id}/{model_type}/{modality}/best.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights {weights_path} do not exist. Please train the model first.")
    print(f"Loading model weights from {weights_path}...")
    test_ds = BottleSeg(root, "test", modality, A_VAL)
    model = get_model(model_type).to(device)

    model.load_state_dict(torch.load(weights_path, weights_only=True))
    print(f"Loaded model path: ../runs/{run_id}/{model_type}/{modality}/best.pt")
    model.eval()

    outdir = pathlib.Path(f"../runs/{run_id}/{model_type}/{modality}/pred/")
    outdir.mkdir(parents=True, exist_ok=True)
    index = 0
    with torch.no_grad():
        total_time = 0
        for img, mask, name in test_ds:
            starting_time = time.time()
            out = model(img.unsqueeze(0).to(device))
            process_time = time.time() - starting_time
            total_time += process_time
            if isinstance(out, dict):
                out = out["logits"]
            pred = (out > 0).squeeze().cpu().numpy().astype("uint8") * 255
            resized = cv2.resize(pred, (1224, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(outdir / f"{name}.png"), resized)
            if index % 100 == 0:
                print(f"Predicted {index} / {len(test_ds)} images...")
            index += 1
        print(f"Mean time {(total_time/index)*1000:.1f} ms per image")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../data")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model_type", choices=list(registry), required=True)
    parser.add_argument("--modality", required=True, choices=["rgb", "dif", "pol"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    predict(args.root, args.run_name, args.modality, args.model_type, args.device)
