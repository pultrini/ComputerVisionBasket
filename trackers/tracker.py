from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np 
import pickle
import os 
import sys

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

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
    
    def draw_elipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame, 
            center=(x_center,y2),
            axes= (int(width), int (0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color= color,
            thickness=2,
            lineType=cv2.LINE_4
            )
        
        rectangle_width =40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2

        y1_rect = (y2-rectangle_width//2) + 15
        y2_rect = (y2+rectangle_width//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text, int(y1_rect+15))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        return frame
        
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _= get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED
        )
        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0,0,0),
            2

        )
        return frame


    def draw_annotations(self, video_frames, tracks):
        output_video_frame = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["person"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw player

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)

            output_video_frame.append(frame)

            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))
        
        return output_video_frame