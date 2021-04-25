from load_json import load_json
import cv2 as cv

config = load_json('config.json')
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")


class PoseEstimation:
    """
    Returns Upper Body pose estimations from openPose
    """

    def __init__(self):
        self.thr = config['thr']
        self.width = config['width']
        self.height = config['height']
        self.inWidth = self.width
        self.inHeight = self.height
        self.BODY_PARTS = config['BODY_PARTS']

    def pose_estimation(self, frame):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        net.setInput(cv.dnn.blobFromImage(frame,
                                          1.0,
                                          (self.inWidth, self.inHeight),
                                          (127.5, 127.5, 127.5),
                                          swapRB=True,
                                          crop=False))
        out_raw = net.forward()
        out = out_raw[:, :19, :, :]
        body_part_indices = list(self.BODY_PARTS.values())

        points = []
        for i in body_part_indices:
            # Slice heatmap of corresponding body's part.
            heatmap = out[0, i, :, :]

            _, conf, _, point = cv.minMaxLoc(heatmap)
            x = (frame_width * point[0]) / out.shape[3]
            y = (frame_height * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > self.thr else None)

        for p in points:
            cv.ellipse(frame, p, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, p, (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = net.getPerfProfile()

        return points
