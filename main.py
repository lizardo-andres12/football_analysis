from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
import cv2


# use to get test image for KNN classifier
def get_player_image(frames, tracks, num_imgs):
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = frames[0]
        
        # get frame pixels from x1 to x2, y1 to y2
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
        num_imgs -= 1

        if not num_imgs:
            break


def main():
    frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(frames,
                              read_from_stub=True,
                              stub_path='stubs/track_stubs.pkl')

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_frames = tracker.draw_annotations(frames, tracks)
    
    save_video(output_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()
