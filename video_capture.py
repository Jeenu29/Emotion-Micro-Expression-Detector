import cv2

def get_video_source(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception("Could not open video source")
    return cap
