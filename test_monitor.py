import cv2
import numpy as np
from mss import mss

# Select first monitor
sct = mss()
monitor = sct.monitors[1]  # [0] is the full virtual screen on macOS, [1] is usually main

print(f"Capturing from monitor: {monitor}")

while True:
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)

    # macOS gives BGRA, convert to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cv2.imshow("Screen Capture macOS", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
