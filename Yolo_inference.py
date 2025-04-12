#%%

from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('input_videos/IMG_9073.mp4',save=True,stream=True)
for result in results:
    print(result)
    print(10*'=')
    for box in result.boxes:
        print(box)