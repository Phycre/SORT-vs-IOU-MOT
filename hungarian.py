import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_batch(detections, trackers):
    """
    Computes IOU between two bounding boxes in the form [x1,y1,x2,y2]
    """
    dets = np.expand_dims(detections, 1)
    trks = np.expand_dims(trackers, 0)

    x1 = np.maximum(dets[..., 0], trks[..., 0])
    y1 = np.maximum(dets[..., 1], trks[..., 1])

    x2 = np.minimum(dets[..., 2], trks[..., 2])
    y2 = np.minimum(dets[..., 3], trks[..., 3])

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    wh = w * h

    det_area = (dets[..., 2] - dets[..., 0]) * (dets[..., 3] - dets[..., 1])
    trk_area = (trks[..., 2] - trks[..., 0]) * (trks[..., 3] - trks[..., 1])

    iou = wh / (det_area + trk_area - wh + 1e-6)

    return iou
    

class HungarianMatcher:
    def solve(self, cost_matrix):
        """
        Use external SciPy linear_sum_assignment API
        """
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        return list(zip(row_idx, col_idx)) #list of (rows, cols) matches


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):

    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    cost_matrix = -iou_matrix

    matcher = HungarianMatcher()
    matched_indices = matcher.solve(cost_matrix)

    matched_indices = np.array(matched_indices)

    unmatched_detections = [
        d for d in range(len(detections))
        if d not in matched_indices[:, 0]
    ]

    unmatched_trackers = [
        t for t in range(len(trackers))
        if t not in matched_indices[:, 1]
    ]

    matches = []
    for d, t in matched_indices:
        if iou_matrix[d, t] < iou_threshold:
            unmatched_detections.append(d)
            unmatched_trackers.append(t)
        else:
            matches.append([d, t])

    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)