from ultralytics import YOLO

DEVICE = 'cuda:0'

model = YOLO('yolo11n-seg.pt')  # load a pretrained model (recommended for training)


# for modality in ['rgb', 'pol', 'dif']:
for modality in ['pol', 'dif']:  # change to 'pol' or 'dif' as needed

    data = f'{modality}.yaml'

    results = model.train(
        data=data,
        epochs=5,
        batch=16,
        imgsz=1024,
        project='run/yolo',      # specify the project name
        name=modality,              # specify the experiment name (folder for logs)
        device=DEVICE,               # specify the GPU device (0 for first GPU)
    )

# metrics = model.val()

