from ultralytics import YOLO
from utils import get_center_of_box, get_width_of_box
from numpy import array
import cv2
import supervision as sv
import pickle
import os


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
            
        detections = self.detect_frames(frames)
        tracks = {
            'players': [],
            'referee': [],
            'ball': []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # change goalkeeper to player
            detection_supervision = sv.Detections.from_ultralytics(detection)
            for obj_i, cls in enumerate(detection_supervision.class_id):
                if cls_names[cls] == 'goalkeeper':
                    detection_supervision.class_id[obj_i] = cls_names_inv['player']
        
            # assign model predicted objects with tracker ids
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls = frame_detection[3]
                track_id = frame_detection[4]

                # add bounding box for each tracked object at the specific frame
                if cls == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                if cls == cls_names_inv['referee']:
                    tracks['referee'][frame_num][track_id] = {'bbox': bbox}

            # ball will not be mistaken for another so no need to track
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls = frame_detection[3]
                
                if cls == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        # tracks will now be populated with bounding box xyxy for every tracked object at every frame
        return tracks

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            
        return detections

    def draw_annotations(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            player_dict = tracks['players'][frame_num]
            ref_dict = tracks['referee'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                color = player.get('team_color', (252, 140, 3))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

            for _, ref in ref_dict.items():
                frame = self.draw_ellipse(frame, ref['bbox'], (119, 252, 3))

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 0, 255))

            output_frames.append(frame)

        return output_frames

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_box(bbox)
        width = get_width_of_box(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
                
            cv2.putText(
                frame,
                f'{track_id}',
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame
            
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_box(bbox)
        
        triangle_points = array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
            