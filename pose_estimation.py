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


    def pose_estimation(self,frame):
        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (self.inWidth, self.inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        self.out_raw = net.forward()
        self.out = self.out_raw[:, :19, :, :]
        self.bodyPartIndices = list(self.BODY_PARTS.values())


        self.points = []
        for i in self.bodyPartIndices:

            # Slice heatmap of corresponding body's part.
            self.heatMap = self.out[0, i, :, :]

            _, self.conf, _, self.point = cv.minMaxLoc(self.heatMap)
            self.x = (self.frameWidth * self.point[0]) / self.out.shape[3]
            self.y = (self.frameHeight * self.point[1]) / self.out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            self.points.append((int(self.x), int(self.y)) if self.conf > self.thr else None)

        for p in self.points:
            cv.ellipse(frame, p, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, p, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        self.t, _ = net.getPerfProfile()
        self.freq = cv.getTickFrequency() / 1000
