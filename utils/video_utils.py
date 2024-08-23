import cv2


def read_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        has_next, frame = cap.read()
        if not has_next:
            break
        frames.append(frame)

    return frames

def save_video(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
    