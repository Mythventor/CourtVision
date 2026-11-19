"""Manual court calibration and camera-pan tracking, on synthetic panning footage."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from courtvision.calibration import (  # noqa: E402
    CalibratedCourtMapper, Calibration, CalibrationSession, CameraTracker,
)
from courtvision.config import load_config, merge  # noqa: E402
from courtvision.court import CourtSpec, landmarks, to_court  # noqa: E402

FW, FH = 1280, 720          # camera frame size
SW, SH = 2600, 900          # scene ("panorama") size the camera pans across


def make_scene(seed=0) -> np.ndarray:
    """Static textured background (walls, floor) with plenty of corners for ORB."""
    rng = np.random.default_rng(seed)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (SH, SW), dtype=np.uint8), (0, 0), 2)
    scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    for _ in range(400):
        x, y = int(rng.integers(0, SW - 60)), int(rng.integers(0, SH - 60))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.rectangle(scene, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 60))), c, -1)
    return scene


def camera(dx: float, zoom: float = 1.0) -> np.ndarray:
    """Scene -> frame homography for a camera panned by dx scene pixels (and zoomed)."""
    cx, cy = SW / 2 + dx, SH / 2
    return np.array([[zoom, 0, FW / 2 - zoom * cx], [0, zoom, FH / 2 - zoom * cy], [0, 0, 1]])


def shoot(scene, T) -> np.ndarray:
    frame = cv2.warpPerspective(scene, T, (FW, FH))
    # burned-in clock and logo: fixed in the image, must not fool the tracker
    cv2.rectangle(frame, (20, FH - 60), (240, FH - 15), (255, 255, 255), -1)
    cv2.putText(frame, "08:21:25", (30, FH - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.rectangle(frame, (FW - 260, FH - 60), (FW - 20, FH - 15), (255, 255, 255), -1)
    return frame


# Scene pixels -> court feet (an arbitrary perspective view of an 84 x 50 ft court)
SCENE_TO_COURT = np.linalg.inv(np.array([[22.0, 6.0, 300.0], [0.0, 9.0, 300.0], [0.0, 0.004, 1.0]]))


def test_presets_and_landmarks():
    hs = CourtSpec.preset("high_school")
    assert (hs.length, hs.width, hs.lane_width) == (84, 50, 12)
    marks = landmarks(hs)
    assert len(marks) == 21
    assert marks["Left free-throw line: far end"] == (19, 19)
    assert marks["Right baseline x near sideline (corner)"] == (84, 50)
    assert CourtSpec.from_config({"standard": "high_school", "length_ft": 74}).length == 74
    with pytest.raises(ValueError):
        CourtSpec.preset("wnba")


def _calibrate(scene, ref_T, names):
    """Clicks (as a user would) the given landmarks on the reference frame."""
    spec = CourtSpec.preset("high_school")
    marks = landmarks(spec)
    court_to_frame = ref_T @ np.linalg.inv(SCENE_TO_COURT)
    session = CalibrationSession("Bulls vs Sonics - Q1.mp4", shoot(scene, ref_T), 60.0,
                                 "high_school")
    for n in names:
        x, y = to_court(np.array([marks[n]], float), court_to_frame)[0]
        session.click(n, x, y)
    return session


NAMES = ["Left lane: baseline corner, far side", "Left lane: baseline corner, near side",
         "Left free-throw line: far end", "Left free-throw line: near end",
         "Center circle: center"]


def test_calibration_maps_frame_to_court():
    scene = make_scene()
    session = _calibrate(scene, camera(-500), NAMES)
    cal = session.calibration()
    assert cal.error_ft() < 0.05                 # exact clicks fit exactly
    # an unclicked landmark lands where it should
    marks = landmarks(cal.court_spec)
    court_to_frame = camera(-500) @ np.linalg.inv(SCENE_TO_COURT)
    corner = to_court(np.array([marks["Left baseline x near sideline (corner)"]], float),
                      court_to_frame)[0]
    H_frame = cal.homography() @ np.diag([1 / FW, 1 / FH, 1])
    np.testing.assert_allclose(to_court(corner[None], H_frame)[0], [0, 50], atol=0.05)


def test_session_save_resume_undo(tmp_path):
    scene = make_scene()
    session = _calibrate(scene, camera(0), NAMES[:3])
    assert "3 of at least 4" in session.status()
    with pytest.raises(ValueError):
        session.save(tmp_path)
    session.click(NAMES[3], 640, 500)
    session.click(NAMES[3], 641, 501)            # re-clicking replaces, not adds
    assert len(session.points) == 4
    session.save(tmp_path)
    resumed, was_saved = CalibrationSession.start("Bulls vs Sonics - Q1.mp4", 0, "nba", tmp_path)
    assert was_saved and len(resumed.points) == 4 and resumed.standard == "high_school"
    resumed.undo()
    assert len(resumed.points) == 3
    assert resumed.image_rgb().shape == (FH, FW, 3)


def test_camera_tracker_follows_pan_and_zoom():
    scene = make_scene()
    ref = shoot(scene, camera(0))
    tracker = CameraTracker(ref, work_width=960)
    pans = [(dx, 1.0 + 0.0004 * dx) for dx in range(0, 601, 30)]     # pan right and zoom in
    probe = np.array([[300.0, 200.0], [900.0, 500.0], [640.0, 360.0]])
    for dx, zoom in pans:
        T = camera(dx, zoom)
        frame = shoot(scene, T)
        M = tracker.update(frame)
        assert M is not None, f"lost at dx={dx}"
        to_norm = tracker.frame_to_normalized(frame.shape, M)
        truth = camera(0) @ np.linalg.inv(T)            # frame -> reference frame (pixels)
        got = to_court(probe, to_norm) * [FW, FH]
        want = to_court(probe, truth)
        assert np.abs(got - want).max() < 3.0, f"dx={dx}: {got} vs {want}"


def test_calibrated_mapper_gives_court_positions_while_panning():
    scene = make_scene()
    ref_T = camera(-300)
    session = _calibrate(scene, ref_T, NAMES)
    mapper = CalibratedCourtMapper(session.calibration(), session.frame)
    player_scene = np.array([[1400.0, 700.0]])           # a player's feet, fixed in the scene
    true_court = to_court(player_scene, SCENE_TO_COURT)[0]
    for dx in range(-300, 151, 25):
        T = camera(dx)
        H = mapper(shoot(scene, T))
        assert H is not None
        feet_px = to_court(player_scene, T)              # where the feet are in this frame
        np.testing.assert_allclose(to_court(feet_px, H)[0], true_court, atol=0.5)


def test_process_video_uses_saved_calibration(tmp_path):
    """A saved court setup is picked up automatically, and redoes stale tracks."""
    from test_video import FakeDetector

    from courtvision.processing import process_video

    scene = make_scene()
    clip = tmp_path / "pan.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (FW, FH))
    for i in range(20):
        writer.write(shoot(scene, camera(-200 + 10 * i)))
    writer.release()
    config = merge(load_config(), {"output_dir": str(tmp_path / "out"),
                                   "video": {"proxy": False},
                                   "render": {"width": 640, "panel_width": 320}})

    first = process_video(clip, config, detector=FakeDetector())
    assert first.tracking.stats["court_mapped"] == 0.0     # no setup yet: no 2D panel

    session = _calibrate(scene, camera(-200), NAMES)
    session.video = clip
    session.frame = cv2.VideoCapture(str(clip)).read()[1]  # the clip's own (compressed) frame
    session.save(config["output_dir"])

    second = process_video(clip, config, detector=FakeDetector())
    assert not second.reused                               # calibration is newer: redone
    assert second.tracking.stats["court_mapped"] > 0.9
    assert second.tracking.court_spec["length"] == 84      # high school court


def test_landmark_choices_and_guide():
    session = CalibrationSession("x.mp4", np.zeros((FH, FW, 3), np.uint8), 0, "high_school")
    choices = session.landmark_choices()
    assert len(choices) == 21 and choices[18] == ("19. Center circle: center", "Center circle: center")
    session.click("Center circle: center", 100, 100)
    assert session.landmark_choices()[18][0].endswith("(placed)")
    guide = session.guide_image_rgb("Right free-throw line: far end")
    assert guide.ndim == 3 and guide.shape[1] == 560
