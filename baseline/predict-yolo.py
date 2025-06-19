from ultralytics import YOLO
import os


# model = YOLO('yolo11n-seg.pt')
model = YOLO('./test_project/test_name/weights/best.pt')


results = model("../data/rgb/images/train/exp07_frame031723.png")  # predict on an image

print(results)  # print results to screen

output_dir = "predicted_images"
os.makedirs(output_dir, exist_ok=True)

for idx, result in enumerate(results):
    output_path = os.path.join(output_dir, f"prediction_{idx}.jpg")
    result.save(filename=output_path)
