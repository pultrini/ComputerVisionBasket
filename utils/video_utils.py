import cv2

def read_videos(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_videos(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def reduce_video_quality(input_video, output_video, reduce_rate=0.5):
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Error to open a video")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_width = int(width*reduce_rate)
    new_heigth = int(heigth*reduce_rate)

    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, codec, fps, (new_width, new_heigth))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        reduce_frame = cv2.resize(frame, (new_width, new_heigth))
        out.write(reduce_frame)
    
    cap.release()
    out.release()
    print("Process conclued.")

    