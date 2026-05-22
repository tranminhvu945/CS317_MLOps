import sys
try:
    import cv2
    print('cv2 OK:', cv2.__version__)
except Exception as e:
    print('cv2 ERROR:', type(e), e)

