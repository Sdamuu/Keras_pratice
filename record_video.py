import cv2
import sys

cap = cv2.VideoCapture(0)

sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mpeg')
# 第三个参数则是镜头快慢的，10为正常，小于10为慢镜头
out = cv2.VideoWriter('./output2.avi', fourcc,fps,sz)
while True:
    ret,frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        a = out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
