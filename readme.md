an AI-based football analysis system using Python, YOLO, and OpenCV for automated analysis
of match footage.
• Applying object detection and tracking techniques to identify and analyze players and extract structured
information from football videos.   
so first of all i take a video from kaggle dataset https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv image = assets\original-video.png  and after that i run the yolo_inference and it gives a video with person and precision with the ball detection rarely done image = assets\video_after_normal_yolo.png  so i want to improve the model so i choose the dataset https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc  and i use it to train my yolo model 