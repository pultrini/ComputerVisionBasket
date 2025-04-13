from ultralytics import YOLO
import supervision as sv
import pickle
import os 


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_objects_track(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {'player': [], 'ball': []}
        for frame_num, detection in enumerate(detections):
            cls_name=detection.names
            cls_name_inv = {v: k for k,v in cls_name.items()}
            print(cls_name)
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks['person'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inv['person']:
                    tracks['person'][frame_num][track_id] = {"bbox":bbox}
                
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_name_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox":bbox}
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)

        return tracks
