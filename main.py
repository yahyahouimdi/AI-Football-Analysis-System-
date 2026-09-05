from utils import read_video, save_video

def main () :
    # Read the video
    video_frames = read_video('input_video/video.mp4')





    #save the video
    save_video(video_frames, 'output_video/output_video.avi')

if __name__ == '__main__':
    main()