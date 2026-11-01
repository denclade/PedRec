import math

import numpy as np

from pedrec.tracking.byte_tracker import ByteTracker, linear_assignment
from pedrec.tracking.one_euro import OneEuroFilter, HumanStateSmoother


def _bb(x, y, w=60, h=160):
    return [x, y, w, h]


def test_bytetrack_keeps_ids_of_moving_persons():
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, new_track_thresh=0.6)
    ids_a, ids_b = set(), set()
    for frame in range(30):
        bbs = np.array([_bb(100 + 4 * frame, 300), _bb(600 - 3 * frame, 320)], dtype=np.float64)
        result = tracker.update(bbs, np.array([0.9, 0.85]))
        ids_a.add(result.assignments[0])
        ids_b.add(result.assignments[1])
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_bytetrack_low_score_and_occlusion_recovery():
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, new_track_thresh=0.6, max_time_lost=10)
    uid = tracker.update(np.array([_bb(100, 300)]), np.array([0.9])).assignments[0]
    # partially occluded: low detection score continues the track
    assert tracker.update(np.array([_bb(104, 300)]), np.array([0.3])).assignments[0] == uid
    # fully occluded for 5 frames
    for _ in range(5):
        tracker.update(np.zeros((0, 4)), np.zeros(0))
    assert tracker.update(np.array([_bb(128, 300)]), np.array([0.9])).assignments[0] == uid


def test_bytetrack_no_new_track_for_low_scores():
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, new_track_thresh=0.6)
    assert tracker.update(np.array([_bb(100, 300)]), np.array([0.3])).assignments == {}


def test_assignment_gates_invalid_pairs_before_optimization():
    # Ungated Hungarian selects both 0.2 entries and discards them, losing the valid 0.1 association.
    matches, unmatched_tracks, unmatched_detections = linear_assignment(np.array([[0.1, 0.2], [0.2, 1.0]]), 0.15)
    assert matches == [(0, 0)]
    assert unmatched_tracks == [1]
    assert unmatched_detections == [1]


def test_bytetrack_duplicate_detections_do_not_create_new_id():
    tracker = ByteTracker(new_track_thresh=0.6)
    uid = tracker.update(np.array([_bb(100, 300)]), np.array([0.9])).assignments[0]
    for frame in range(10):
        x = 100 + frame
        # Lower-scoring duplicate comes first: assignments must preserve original detection indices.
        result = tracker.update(np.array([_bb(x + 1, 300), _bb(x, 300)]), np.array([0.8, 0.95]), ['dup', 'real'])
        assert result.assignments == {1: uid}
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].payload == 'real'
    assert tracker.next_uid == 1


def test_bytetrack_keeps_two_distinct_overlapping_persons():
    tracker = ByteTracker(new_track_thresh=0.6)
    for frame in range(10):
        result = tracker.update(np.array([_bb(100 + frame, 300), _bb(125 + frame, 300)]), np.array([0.9, 0.8]))
        assert result.assignments == {0: 0, 1: 1}


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    signal = 100 + rng.normal(0, 5, size=300)
    filtered = np.array([f(np.array([x]), 1 / 30)[0] for x in signal])
    assert filtered[50:].std() < signal[50:].std() / 2


def test_orientation_smoothing_handles_wrap_around():
    smoother = HumanStateSmoother()
    phis = [math.radians(a) for a in [355, 358, 1, 4, 7]]
    for phi in phis:
        out = smoother.smooth_orientation(np.array([[math.pi / 2, phi], [math.pi / 2, phi]]), 1 / 30)
    # a linear filter on the raw angle would end up around 180 degrees
    assert math.degrees(out[0, 1]) > 340 or math.degrees(out[0, 1]) < 10
    assert 0 <= out[0, 1] < 2 * math.pi
