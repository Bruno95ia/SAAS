import cv2
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE = cv2.CascadeClassifier(_CASCADE_PATH)
def maybe_blur(frame, enabled=True):
    if not enabled or _FACE.empty():
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    out = frame.copy()
    for (x,y,w,h) in faces:
        roi = out[y:y+h, x:x+w]
        if roi.size == 0: continue
        out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (51,51), 0)
    return out
