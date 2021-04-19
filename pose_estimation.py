from load_json import load_json
import cv2 as cv
config = load_json('config.json')
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

class PoseEstimation:
    """
    Pose Estimation
    """
    def __init__(self):
        self.thr = config['thr']
        self.width = config['width']
        self.height = config['height']
        self.inWidth = self.width
        self.inHeight = self.height
        self.BODY_PARTS = config['BODY_PARTS']
        self.POSE_PAIRS = config['POSE_PAIRS']


    def pose_estimation(self,frame):
        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (self.inWidth, self.inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        self.out_raw = net.forward()
        self.out = self.out_raw[:, :19, :, :]

        assert (len(self.BODY_PARTS) == self.out.shape[1])

        self.points = []
        for i in range(18):#range(len(self.BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            self.heatMap = self.out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, self.conf, _, self.point = cv.minMaxLoc(self.heatMap)
            self.x = (self.frameWidth * self.point[0]) / self.out.shape[3]
            self.y = (self.frameHeight * self.point[1]) / self.out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            self.points.append((int(self.x), int(self.y)) if self.conf > self.thr else None)

        for pair in self.POSE_PAIRS:
            self.partFrom = pair[0]
            self.partTo = pair[1]
            assert (self.partFrom in self.BODY_PARTS)
            assert (self.partTo in self.BODY_PARTS)

            self.idFrom = self.BODY_PARTS[self.partFrom]
            self.idTo = self.BODY_PARTS[self.partTo]

            if self.points[self.idFrom] and self.points[self.idTo]:
                cv.line(frame, self.points[self.idFrom], self.points[self.idTo], (0, 255, 0), 3)
                cv.ellipse(frame, self.points[self.idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, self.points[self.idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        self.t, _ = net.getPerfProfile()
        self.freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (self.t / self.freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        return frame