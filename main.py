from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2

prototxt="MobileNetSSD_deploy.prototxt"
model="MobileNetSSD_deploy.caffemodel"
dconfidence=0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Output/output.avi', fourcc, 20.0, (1280, 720))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)
fps = FPS().start()
    
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # out.release()

        if confidence > dconfidence:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            

            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    image = cv2.resize(frame, (1280,720))
    out.write(image)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()
    
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
vs.stream.release()
out.release()