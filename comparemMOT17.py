### NOTE MM requires numpy < 2.0; known to work with 1.26.4
### pip install numpy==1.26.4
###(might make other stuff you work on break as newest opencv.. etc require <= 2.0)
import os
import numpy as np
import motmetrics as mm
from sort import Sort

def load_mot17_sequence(seq_path):
    det_path = seq_path + "/det/det.txt"
    gt_path = seq_path + "/gt/gt.txt"

    dets = np.loadtxt(det_path, delimiter=",")
    gt = np.loadtxt(gt_path, delimiter=",")
    #get max from from dets or gt
    max_frame = int(max(np.max(dets[:, 0]), np.max(gt[:, 0])))

    return dets, gt, max_frame

def run_sort_tracker(detections_by_frame):
    tracker = Sort()
    results = []
    for frame in sorted(detections_by_frame.keys()):
        dets_xywh = detections_by_frame[frame]

        dets = []
        for d in dets_xywh:
            x, y, w, h = d 
            dets.append([x, y, x + w, y + h])
        dets = np.asarray(dets)

        tracks = tracker.update(dets)
        for t in tracks:
            x1, y1, x2, y2, tid = t 
            w = x2 - x1
            h = y2 - y1 

            results.append([
                frame,
                int(tid),
                x1, y1, w, h, 
                1, 1, 1
                ])
    return results



def run_iou_tracker(detections_by_frame):
    return 0
    #run iou here
    #return tracking results in MOTChallenge format


def save_results(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, results, delimiter=",")
    print(f"saved results to {out_path}")


#use mm to compare already done mot
def evaluate_mot(gt_path, result_path):
    gt = mm.io.loadtxt(gt_path)
    res = mm.io.loadtxt(result_path)

    mh = mm.metrics.create()
    acc = mm.utils.compare_to_groundtruth(
            gt, res, distth=0.5
    )
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics)
    print("\nEvaluation Results:")
    print(summary)
    return summary

def main():
    seq = "train/MOT17-02-SDP"
    data_root = "MOT17/"

    seq_path = os.path.join(data_root, seq)

    dets, gt, max_frame = load_mot17_sequence(seq_path)

    detections_by_frame = {}
    for frame in range(1, max_frame + 1):
        #from dets splice data from this frame
        frame_dets = dets[dets[:, 0] == frame][:, 2:6] # x, y, w, h
        detections_by_frame[frame] = frame_dets

    sort_results = run_sort_tracker(detections_by_frame)
    sort_outfile = f"results/{seq}_SORT.txt"
    save_results(sort_results, sort_outfile)

 #   iou_results = run_iou_tracker(detections_by_frame)
 #   iou_outfile = f"results/{seq}_iou.txt"
 #   save_results(iou_results, iou_outfile)


    #finish
    gt_path = os.path.join(seq_path, "gt", "gt.txt")

    print("\nEvaluating SORT")
    evaluate_mot(gt_path, sort_outfile)

 #   print("\nEvaluating IOU")
 #   evaluate_mot(gt_path, iou_outfile)

if __name__ == "__main__":
    main()

