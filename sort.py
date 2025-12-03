import numpy as np
from kalmanTracker import KalmanBoxTracker
from hungarian import associate_detections_to_trackers

class Sort:
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

        self.frame_count = 0 

    def update(self, dets):
        self.frame_count += 1

        trks = []
        to_del = []
        for i, t in enumerate(self.trackers):
            pos = t.predict()
            if np.any(np.isnan(pos)):
                to_del.append(i)
            trks.append(pos.reshape(4))
        trks = np.array(trks)

        new_trackers = []
        for i, t in enumerate(self.trackers):
            if i not in to_del:
                new_trackers.append(t)
        self.trackers = new_trackers


        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched:
            det_idx, trk_idx = m 
            self.trackers[trk_idx].update(dets[det_idx])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        ret = []
        for trk in self.trackers():
            d = trk.get_state().reshape(4)
            if(trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])))
        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]
            
        if len(ret) > 0:
            return np.stack(ret)
        return np.empty((0, 5))


