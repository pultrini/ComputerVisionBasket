from utils import read_videos, save_videos, reduce_video_quality
from trackers import Tracker

def main():
    #reduce video quality 
    #reduce_video_quality('input_videos/IMG_9073.mp4', 'output_videos/output_video.avi')

    #read videos
    video_frames = read_videos('input_videos/input_video.avi')

    #Initialize Tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_objects_track(video_frames, read_from_stub=True, stub_path='stubs/tracks_stubbs.pk1')

    #Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #save video
    save_videos(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()