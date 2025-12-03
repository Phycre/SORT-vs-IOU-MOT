import numpy as np


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

    iou = wh / (det_area + trk_area - wh)

    return iou
    

class HungarianMatcher:
    def _step1_2_reduce(self, C):
        C = C.copy().astype(float)
        n = C.shape[0]

        
        # Subtract the minimum of each row 
        for i in range(n):
            C[i] -= C[i].min()

        # Subtract the minimum of each column
        for j in range(n):
            C[:, j] -= C[:, j].min()

        return C
        
    def _cover_zeros(self, C):
        """
        Cover all zeros using the minimum number of rows and columns
        """
        n = C.shape[0]
        zero_pos = np.where(C == 0)

        row_zero_count = {i: list(zero_pos[0]).count(i) for i in range(n)}
        col_zero_count = {j: list(zero_pos[1]).count(j) for j in range(n)}

        covered_rows = set()
        covered_cols = set()

        while zero_pos[0].size > 0:
            max_row = max(row_zero_count, key=row_zero_count.get)
            max_col = max(col_zero_count, key=col_zero_count.get)

            if row_zero_count[max_row] >= col_zero_count[max_col]:
                covered_rows.add(max_row)
                for j in range(n):
                    if C[max_row, j] == 0:
                        col_zero_count[j] -= 1
                row_zero_count[max_row] = 0
            else:
                covered_cols.add(max_col)
                for i in range(n):
                    if C[i, max_col] == 0:
                        row_zero_count[i] -= 1
                col_zero_count[max_col] = 0

            zero_pos = np.where(C == 0)

        return covered_rows, covered_cols

    def _adjust_matrix(self, C, covered_rows, covered_cols):
        """
        If total coverage doesn't equal to matrix size, we need to adjust matrix
        """

        n = C.shape[0]

        # mask = True means uncovered area
        mask = np.ones_like(C, dtype=bool)
        mask[list(covered_rows), :] = False
        mask[:, list(covered_cols)] = False

        m = C[mask].min()

        C[mask] -= m

        for r in covered_rows:
            for c in covered_cols:
                C[r, c] += m

        return C

    def _extract_assignment(self, C):
        n = C.shape[0]
        assigned_rows = set()
        assigned_cols = set()
        assignments = []

        for _ in range(n):
            # Assign rows that contain exactly one zero
            for i in range(n):
                if i in assigned_rows:
                    continue
                zero_cols = np.where(C[i] == 0)[0]
                zero_cols = [c for c in zero_cols if c not in assigned_cols]
                if len(zero_cols) == 1:
                    c = zero_cols[0]
                    assignments.append((i, c))
                    assigned_rows.add(i)
                    assigned_cols.add(c)

            # Assign columns that contain exactly one zero
            for j in range(n):
                if j in assigned_cols:
                    continue
                zero_rows = np.where(C[:, j] == 0)[0]
                zero_rows = [r for r in zero_rows if r not in assigned_rows]
                if len(zero_rows) == 1:
                    r = zero_rows[0]
                    assignments.append((r, j))
                    assigned_rows.add(r)
                    assigned_cols.add(j)

        return assignments

    def solve(self, cost_matrix):
        """
        main function of Hungarian tracker
        """
        C = self._step1_2_reduce(cost_matrix)
        n = C.shape[0]

        while True:
            covered_rows, covered_cols = self._cover_zeros(C)
            if len(covered_rows) + len(covered_cols) == n:
                break
            C = self._adjust_matrix(C, covered_rows, covered_cols)

        return self._extract_assignment(C) #list of (rows, cols) matches


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