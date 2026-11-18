"""Video pipeline tests on a synthetic clip with fake models (no GPU or Roboflow needed)."""

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sv = pytest.importorskip("supervision")

from courtvision.config import load_config, merge  # noqa: E402
from courtvision.court import CourtCanvas, CourtSpec, homography_from_keypoints, to_court  # noqa: E402
from courtvision.detection import split_roles  # noqa: E402
from courtvision.library import list_videos  # noqa: E402
from courtvision.pipeline import TrackingResult  # noqa: E402
from courtvision.possession import FrameRecord  # noqa: E402
from courtvision.processing import process_video  # noqa: E402
from courtvision.render import fill_ball_gaps, interpolate  # noqa: E402

W, H, FPS, N = 320, 240, 10, 40
RED, BLUE, ORANGE = (0, 0, 220), (220, 0, 0), (0, 140, 255)


def make_clip(path: Path) -> Path:
    """Two 'players' (red, blue) crossing the frame and an orange 'ball'."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        img = np.full((H, W, 3), 128, np.uint8)
        cv2.rectangle(img, (20 + 4 * i, 100), (50 + 4 * i, 160), RED, -1)
        cv2.rectangle(img, (270 - 4 * i, 60), (300 - 4 * i, 120), BLUE, -1)
        if i % 5 != 3:  # ball missing on some frames
            cv2.circle(img, (160, 30 + 3 * i), 5, ORANGE, -1)
        writer.write(img)
    writer.release()
    return path


class FakeDetector:
    """Finds the colored shapes by color, like a detector would."""

    classes = {"player": ["player"], "ball": ["ball"], "hoop": ["rim"], "referee": ["referee"]}

    def __call__(self, frame):
        boxes, names = [], []
        for color, name in ((RED, "player"), (BLUE, "player"), (ORANGE, "ball")):
            mask = cv2.inRange(frame, np.clip(np.array(color) - 40, 0, 255),
                               np.clip(np.array(color) + 40, 0, 255))
            ys, xs = np.nonzero(mask)
            if len(xs) > 10:
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
                names.append(name)
        dets = sv.Detections(
            xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
            confidence=np.full(len(boxes), 0.9),
            class_id=np.zeros(len(boxes), dtype=int),
            data={"class_name": np.asarray(names)},
        )
        return split_roles(dets, self.classes)


# Image pixels -> court feet: the whole frame is the whole court
H_FAKE = np.diag([94 / W, 50 / H, 1.0])


@pytest.fixture
def config(tmp_path):
    return merge(load_config(), {
        "output_dir": str(tmp_path / "outputs"),
        "video": {"frame_stride": 2},
        "court": {"every_n": 1, "template": [[0, 0], [94, 0], [94, 50], [0, 50]]},
        "render": {"width": 320, "panel_width": 240},
    })


def test_process_video_end_to_end(tmp_path, config):
    clip = make_clip(tmp_path / "game_q1.mp4")
    result = process_video(clip, config, detector=FakeDetector(), court_mapper=lambda f: H_FAKE)

    tr = result.tracking
    assert len(tr.frames) == N // 2
    assert tr.stats["tracks"] == 2
    assert sorted(tr.clusters.values()) == [0, 1]        # red and blue on different teams
    assert tr.stats["court_mapped"] == 1.0
    assert 0.5 < tr.stats["ball_found"] < 1.0

    # cache round trip
    loaded = TrackingResult.load(result.folder / "tracks.json.gz")
    assert len(loaded.frames) == len(tr.frames) and loaded.clusters == tr.clusters
    np.testing.assert_allclose(loaded.frames[3].court_H, H_FAKE)

    # rendered video has every source frame, at the output size
    cap = cv2.VideoCapture(str(result.video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    assert ok and frames >= N - 2
    assert first.shape[1] == 320 + 240

    # second run reuses the saved tracks
    again = process_video(clip, config, detector=FakeDetector(), court_mapper=lambda f: H_FAKE)
    assert again.reused


def test_time_range_and_job_folder(tmp_path, config):
    clip = make_clip(tmp_path / "clip.mp4")
    result = process_video(clip, config, start_s=1.0, end_s=3.0,
                           detector=FakeDetector(), court_mapper=lambda f: None)
    assert result.folder.name == "clip_1-3"
    assert all(1.0 <= f.t < 3.0 for f in result.tracking.frames)
    assert result.tracking.stats["court_mapped"] == 0.0


def test_homography_from_keypoints_recovers_mapping():
    true_H = np.array([[0.2, 0.01, -5], [0.0, 0.25, -3], [0.0, 0.0005, 1]])
    court = np.array([[0, 0], [94, 0], [94, 50], [0, 50], [47, 25], [19, 17]], float)
    image = to_court(court, np.linalg.inv(true_H))  # project court points into the image
    template = [p.tolist() for p in court] + [None]
    xy = np.vstack([image, [[1, 1]]])
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.99])  # one low-confidence point
    H_est = homography_from_keypoints(xy, conf, template, 0.5)
    np.testing.assert_allclose(to_court(image, H_est), court, atol=1e-3)
    assert homography_from_keypoints(xy[:3], conf[:3], template, 0.5) is None


def test_interpolate_between_analyzed_frames():
    a = FrameRecord(frame=0, t=0, player_xyxy=np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float),
                    player_track_ids=np.array([1, 2]), ball_xyxy=None, hoop_xyxy=np.zeros((0, 4)))
    b = FrameRecord(frame=4, t=0.4, player_xyxy=np.array([[8, 0, 18, 10]], float),
                    player_track_ids=np.array([1]), ball_xyxy=None, hoop_xyxy=np.zeros((0, 4)))
    s = interpolate(a, b, np.array([0.0, 0.0]), np.array([4.0, 8.0]), 1)
    assert s.ids == [1, 2]                      # track 2 kept until halfway
    np.testing.assert_allclose(s.boxes[0], [2, 0, 12, 10])
    np.testing.assert_allclose(s.ball, [1, 2])
    assert interpolate(a, b, None, None, 3).ids == [1]


def test_fill_ball_gaps():
    def rec(ball):
        return FrameRecord(frame=0, t=0, player_xyxy=np.zeros((0, 4)),
                           player_track_ids=np.zeros(0, int),
                           ball_xyxy=None if ball is None else np.array(ball, float),
                           hoop_xyxy=np.zeros((0, 4)))

    frames = [rec([0, 0, 2, 2]), rec(None), rec([4, 4, 6, 6]), rec(None), rec(None), rec(None),
              rec([0, 0, 2, 2])]
    filled = fill_ball_gaps(frames, max_gap=2)
    np.testing.assert_allclose(filled[1], [3, 3])
    assert filled[3] is None                    # gap of 3 is too long to fill


def test_court_canvas_draws_at_panel_size():
    canvas = CourtCanvas(CourtSpec(), 480)
    img = canvas.draw()
    assert img.shape == (canvas.height, 480, 3)
    assert canvas.px(0, 0) == (16, 16)
    assert canvas.inside(47, 25) and not canvas.inside(120, 25)


def test_list_videos_groups_by_folder(tmp_path):
    for p in ["Bulls vs Jazz/q1.mp4", "Bulls vs Jazz/q2.MOV", "Bulls vs Jazz/notes.txt",
              "Lakers vs Spurs/q1.mp4", "outputs/q1_0-end/annotated.mp4"]:
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_bytes(b"")
    videos = list_videos(tmp_path, exclude=tmp_path / "outputs")
    assert {g: [p.name for p in ps] for g, ps in videos.items()} == {
        "Bulls vs Jazz": ["q1.mp4", "q2.MOV"],
        "Lakers vs Spurs": ["q1.mp4"],
    }
    assert list_videos(tmp_path / "missing") == {}
