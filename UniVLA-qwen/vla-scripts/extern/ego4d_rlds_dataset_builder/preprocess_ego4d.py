import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Ego4D video clips into frame sequences.')
    parser.add_argument('--denseclips_dir', type=str, required=False,
                        default='/fs/scratch/PAS2099/danielf/geometry_grounded_latents/data_prep/ego4d/post')
    parser.add_argument('--info_clips_json', type=str, required=False,
                        default='/fs/scratch/PAS2099/danielf/geometry_grounded_latents/UniVLA/vla-scripts/extern/ego4d_rlds_dataset_builder/univla-ego4d-rlds-dependencies/info_clips.json')
    parser.add_argument('--source_videos_dir', type=str, required=False,
                        default='/fs/scratch/PAS2099/danielf/geometry_grounded_latents/data_prep/ego4d/v2/clips')
    parser.add_argument('--frame_interval', type=int, default=15)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip writing frames that already exist.')
    return parser.parse_args()

def _worker_init():
    # Prevent OpenCV from spawning threads inside each worker
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def _compute_targets_per_clip(video_name, clips, frame_interval, denseclips_dir):
    """
    Returns:
      targets_by_frame: dict[int, list[(save_path, save_idx)]]
      annotations: list[dict]
    """
    # Sort clips by start frame for stable, single-pass traversal
    clips_sorted = sorted(clips, key=lambda c: c['pre_frame']['frame_num'])

    targets_by_frame = defaultdict(list)
    annotations = []

    for idx, clip in enumerate(clips_sorted):
        start = int(clip['pre_frame']['frame_num'])
        end   = int(clip['post_frame']['frame_num'])

        # Action directory
        action_name = clip['pre_frame']['path'].split('/')[1]
        save_dir = os.path.join(denseclips_dir, video_name, action_name)
        os.makedirs(save_dir, exist_ok=True)

        # Exact frame indices to save (include end even if not on interval)
        # NB: frame numbers are absolute in the source video
        sel = list(range(start, end + 1, frame_interval))
        if sel[-1] != end:
            sel.append(end)

        # Map each absolute frame index -> (save_path, save_idx)
        # save_idx is 1-based, consistent with your original naming
        for i, fidx in enumerate(sel, start=1):
            npy_path = os.path.join(save_dir, f"{i:05d}.npy")
            targets_by_frame[fidx].append((npy_path, i))

        annotations.append({
            'video_name': video_name,
            'action_name': action_name,
            'source_video': None,  # fill later once we know full path
            'start_frame': start,
            'end_frame': end,
            'language': clip['narration_text'],
            'id': idx,
        })

    return targets_by_frame, annotations

def _process_one_video(args_tuple):
    video_name, clips, args = args_tuple
    video_path = os.path.join(args.source_videos_dir, f"{video_name}.mp4")

    targets_by_frame, annotations = _compute_targets_per_clip(
        video_name, clips, args.frame_interval, args.denseclips_dir
    )
    for a in annotations:
        a['source_video'] = video_path

    # Nothing to do?
    if not targets_by_frame:
        return annotations

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Return annotations anyway; caller can log failures
        return annotations

    # We do a single linear scan. OpenCV decoding is fastest when sequential.
    # Grab total frames so we can early-exit.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max(targets_by_frame) + 1

    # Precompute the *sorted* unique target frames and use a pointer to skip work
    target_frames_sorted = sorted(targets_by_frame.keys())
    ptr = 0
    next_target = target_frames_sorted[ptr]

    # Fast-forward by grab() until we reach the first target
    current_idx = -1
    while True:
        ret = cap.grab()
        if not ret:
            break
        current_idx += 1

        # Skip until we reach the next frame we actually need to decode
        if current_idx < next_target:
            continue

        # We need this frame: retrieve it
        ret, frame = cap.retrieve()
        if not ret:
            break

        # Write all outputs for this frame (can be multiple clips overlapping)
        for npy_path, _save_idx in targets_by_frame[current_idx]:
            if args.skip_existing and os.path.exists(npy_path):
                continue
            # np.save releases the GIL during write; OK in multiprocess
            np.save(npy_path, frame)

        # Advance target pointer
        ptr += 1
        if ptr >= len(target_frames_sorted):
            break
        next_target = target_frames_sorted[ptr]

    cap.release()
    return annotations

def main():
    args = parse_arguments()
    os.makedirs(args.denseclips_dir, exist_ok=True)

    with open(args.info_clips_json, 'r') as f:
        clip_data = json.load(f)

    jobs = [(video_name, clips, args) for video_name, clips in clip_data.items()]

    all_annotations = []
    if args.processes > 1:
        with Pool(processes=args.processes, initializer=_worker_init) as pool:
            for ann in tqdm(pool.imap_unordered(_process_one_video, jobs), total=len(jobs), desc="Processing videos"):
                all_annotations.extend(ann)
    else:
        _worker_init()
        for job in tqdm(jobs, desc="Processing videos"):
            all_annotations.extend(_process_one_video(job))

    with open(os.path.join(args.denseclips_dir, 'annotations.json'), 'w') as f:
        json.dump(all_annotations, f, indent=4)

if __name__ == '__main__':
    main()
