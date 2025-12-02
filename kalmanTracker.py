import numpy as np



def convert_bbox_to_z(bbox):
    """
    [x1, y1, x2, y2] to [x, y, s, r]
    """

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h                # area
    r = w / float(h)         # w to h ratio 
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    takes a bounding box of [x, y, s, r] to [x1, y1, x2, y2] topleft to bottom right
    """

    w = np.sqrt(x[2] * x[3])    # x[3] = w to h ratio
    h = x[2] / w                # x[2] = area

    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0

    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))
    

class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox):
        # x = [u, v, s, r, u_dot, v_dot, s_dot] ; z = [u, v, s, r]
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history = [] 


        self.x = np.zeros((7, 1))
        self.x[:4] = convert_bbox_to_z(bbox)

        # initialize matrix F, H, P, R, Q
        self.F = np.eye(7)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1
        self.H = np.eye(4, 7)
        self.P = np.diag([10, 10, 10, 10, 100, 100, 100]).astype(float)
        # R(Measurement Noise)
        self.R = np.diag([10, 10, 40, 40]).astype(float)
        # Q (Process Noise)
        self.Q = np.eye(7) * 0.01
        self.Q[4:, 4:] = 0.0001

    def update(self, bbox):
        """
        correct bias from predict()
        K = PH^T * inv(HPH^T + R)
        x = x + K(z - Hx)
        P = (I - KH)P
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        z = convert_bbox_to_z(bbox)
        # matrices mult
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.x.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)


    def predict(self):
        """
        predict the next FPS
        x = Fx
        P = FPF^T + Q
        """
        # prevent area becomes negative
        if((self.x[6]+self.x[2])<=0):
            self.x[6] *= 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # maintain age var
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(convert_x_to_bbox(self.x))
        return self.history[-1]
    
    def get_state(self):
        return convert_x_to_bbox(self.x)
    