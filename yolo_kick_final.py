#!/usr/bin/env python3
# coding=utf-8
import argparse
import importlib
import importlib.util
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / 'TonyPi').exists():
            return candidate
    return start.resolve().parent


def bootstrap_paths(repo_root: Path) -> None:
    tonypi_root = repo_root / 'TonyPi'
    sdk_root = tonypi_root / 'HiwonderSDK'
    for path in (repo_root, tonypi_root, sdk_root):
        value = str(path)
        if path.exists() and value not in sys.path:
            sys.path.insert(0, value)


def resolve_default_model(repo_root: Path, model_arg: str) -> Path:
    if model_arg:
        model = Path(model_arg)
        return model if model.is_absolute() else (repo_root / model)
    candidates = [
        repo_root / 'models' / 'best.pt',
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


@dataclass
class Detection:
    name: str
    conf: float
    cx: float
    cy: float
    xyxy: Tuple[int, int, int, int]


class CameraAdapter:
    def __init__(self, source: str):
        self.mode = 'cv2'
        self.cap = None
        self.robot_camera = None
        if source == 'robot':
            try:
                import hiwonder.Camera as HiwonderCamera
                self.robot_camera = HiwonderCamera.Camera()
                self.robot_camera.camera_open()
                self.mode = 'robot'
            except Exception as exc:
                raise RuntimeError(f'open robot camera failed: {exc}')
        else:
            parsed = int(source) if source.lstrip('-').isdigit() else source
            self.cap = cv2.VideoCapture(parsed)
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open camera source: {source}')

    def read(self):
        if self.mode == 'robot':
            return self.robot_camera.read()
        return self.cap.read()

    def close(self):
        if self.mode == 'robot' and self.robot_camera is not None:
            self.robot_camera.camera_close()
        elif self.cap is not None:
            self.cap.release()


class MotionAdapter:
    def __init__(self, run_on_robot: bool):
        self.run_on_robot = run_on_robot
        self.available = False
        self.agc = None
        self._connect()

    def _connect(self):
        if not self.run_on_robot:
            return
        import_errors = []
        module_candidates = [
            'hiwonder.ActionGroupControl',
            'TonyPi.HiwonderSDK.hiwonder.ActionGroupControl',
        ]
        for module_name in module_candidates:
            try:
                self.agc = importlib.import_module(module_name)
                self.available = True
                print(f'[motion] using module: {module_name}')
                return
            except Exception as exc:
                import_errors.append(f'{module_name}: {exc}')
        print('[motion] import failed, fallback to dry-run')
        for item in import_errors:
            print(f'[motion]  - {item}')
        self.run_on_robot = False

    def run_action(self, action: str, times: int = 1):
        if times <= 0:
            return
        if self.available and self.agc is not None:
            self.agc.runActionGroup(action, times=times)
        else:
            print(f'[dry-run] action={action} times={times}')

    def stand(self):
        self.run_action('stand_slow', times=1)


def set_head_pose_for_pre_approach(duration_ms: int = 300):
    """Match approachball head limits: vertical=min(servo1), horizontal=center(servo2)."""
    try:
        import hiwonder.ros_robot_controller_sdk as rrc
        import hiwonder.yaml_handle as yaml_handle
        from hiwonder.Controller import Controller

        servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        servo1_min = int(servo_data.get('servo1', 1000))
        servo2_center = int(servo_data.get('servo2', 1500))
        board = rrc.Board()
        ctl = Controller(board)
        ms = max(100, int(duration_ms))
        ctl.set_pwm_servo_pulse(1, servo1_min, ms)
        ctl.set_pwm_servo_pulse(2, servo2_center, ms)
        time.sleep(ms / 1000.0 + 0.05)
        print(f'[pre-approach] head pose set: servo1(min)={servo1_min}, servo2(center)={servo2_center}')
        return True
    except Exception as exc:
        print(f'[pre-approach] head pose setup skipped: {exc}')
        return False


def tilt_head_up_step(ctx: "SharedContext", step: int = 20, duration_ms: int = 120):
    """Raise vertical head servo by a small step when goal is missing."""
    try:
        import hiwonder.ros_robot_controller_sdk as rrc
        import hiwonder.yaml_handle as yaml_handle
        from hiwonder.Controller import Controller

        with ctx.lock:
            current = ctx.head_servo1_pos

        servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        if current is None:
            current = int(servo_data.get('servo1', 1000))
        target = int(max(500, min(2500, int(current) + int(step))))

        board = rrc.Board()
        ctl = Controller(board)
        ms = max(80, int(duration_ms))
        ctl.set_pwm_servo_pulse(1, target, ms)
        time.sleep(ms / 1000.0)

        with ctx.lock:
            ctx.head_servo1_pos = target
        return True
    except Exception as exc:
        print(f'[head] tilt up skipped: {exc}')
        return False


class SharedContext:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = 'detect_ball_color'
        self.status_text = 'waiting ball color'
        self.latest_ball: Optional[Detection] = None
        self.latest_goal: Optional[Detection] = None
        self.center_x = 320
        self.ball_rgb_text = ''
        self.approach_started = False
        self.approach_done = False
        self.approach_returncode: Optional[int] = None
        self.kick_done = False
        self.vision_ready_after_approach = False
        self.dynamic_round = 0
        self.pre_side_move_dir = 'none'
        self.head_servo1_pos = None
        self.exit_requested = False
        self.program_start_ts: Optional[float] = None


class ApproachBallSession:
    """Run approachball.py as an in-process module while keeping its own worker thread."""

    def __init__(self, approach_module):
        self.mod = approach_module
        self.started = False
        self.done = False

    def start(self, target_rgb_text: str, align_round: int, debug: bool = False) -> None:
        self.mod.debug = bool(debug)
        self.mod.align_round = int(align_round)
        self.mod.init()
        self.mod.reset()
        self.mod.set_target_color_from_rgb(self.mod.parse_rgb_text(target_rgb_text))
        self.mod.start()
        self.started = True
        self.done = False

    def step(self, frame):
        if not self.started:
            return frame, False
        view = self.mod.run(frame.copy())
        finished = bool(getattr(self.mod, 'kick_ready_to_exit', False))
        if finished and not self.done:
            self.done = True
            try:
                self.mod.stop()
            except Exception:
                pass
        return view, self.done

    def abort(self) -> None:
        if not self.started:
            return
        try:
            self.mod.stop()
        except Exception:
            pass


def load_approach_module(approach_script: Path):
    module_name = f'approachball_merge_{approach_script.stem}'
    spec = importlib.util.spec_from_file_location(module_name, str(approach_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module spec from {approach_script}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Dual-thread YOLO flow: display in main thread; post-approach align-and-shoot in worker thread.'
    )
    parser.add_argument('--model', type=str, default='', help='YOLO model path, default auto find best.pt')
    parser.add_argument('--camera', type=str, default='robot', help='camera source: robot/0/1/video path')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='YOLO infer size')
    parser.add_argument('--device', type=str, default='', help='YOLO device, e.g. cpu/0')
    parser.add_argument('--ball-class', type=str, default='redball', help='ball class name in YOLO model')
    parser.add_argument('--goal-class', type=str, default='goal', help='goal class name in YOLO model')
    parser.add_argument('--disable-color-fallback', action='store_true', help='disable HSV color fallback when YOLO misses')
    parser.add_argument('--stable-frames', type=int, default=6, help='frames used to average ball RGB')
    parser.add_argument('--roi-scale', type=float, default=0.45, help='center ROI scale in bbox for color extraction')
    parser.add_argument('--approach-script', type=str, default='approachball.py', help='approachball script path')
    parser.add_argument('--python', type=str, default=sys.executable, help='reserved compatibility arg (unused in merged mode)')
    parser.add_argument('--show', action='store_true', help='show preview window')
    parser.add_argument('--dry-run-subprocess', action='store_true', help='skip approach stage and jump to align+shoot')
    parser.add_argument('--run-on-robot', action='store_true', help='execute real action groups')
    parser.add_argument('--force-dry-run-actions', action='store_true', help='force dry-run for align+shoot actions')

    parser.add_argument('--line-align-tol', type=int, default=35, help='|ball.cx-goal.cx| tolerance')
    parser.add_argument('--center-align-tol', type=int, default=40, help='line center to image center tolerance')
    parser.add_argument('--goal-face-tol', type=int, default=40, help='goal center to image center tolerance')
    parser.add_argument('--ball-hold-tol', type=int, default=70, help='ball center to image center tolerance for shot')
    parser.add_argument('--action-cooldown', type=float, default=0.14, help='min interval between actions')

    parser.add_argument('--turn-left-action', type=str, default='turn_left_small_step', help='left turn action')
    parser.add_argument('--turn-right-action', type=str, default='turn_right_small_step', help='right turn action')
    parser.add_argument('--left-move-action', type=str, default='left_move_fast', help='left move action')
    parser.add_argument('--right-move-action', type=str, default='right_move_fast', help='right move action')
    parser.add_argument('--left-shot-action', type=str, default='left_shot', help='left shot action')
    parser.add_argument('--right-shot-action', type=str, default='right_shot', help='right shot action')

    parser.add_argument('--turn-times', type=int, default=1, help='times for turn action')
    parser.add_argument('--move-times', type=int, default=1, help='times for side move action')
    parser.add_argument('--search-turn-times', type=int, default=2, help='times for search turn when target missing')
    parser.add_argument('--pre-center-tol', type=int, default=15, help='ball center tolerance before approachball')
    parser.add_argument('--pre-turn-px', type=float, default=55.0, help='approx px represented by one in-place turn')
    parser.add_argument('--pre-extra-turns', type=int, default=2, help='extra same-direction turns for kick margin')
    parser.add_argument('--pre-center-forward-action', type=str, default='go_forward_fast', help='forward action when ball is already near center')
    parser.add_argument('--pre-center-forward-times', type=int, default=1, help='forward times when ball is already near center')
    parser.add_argument('--pre-reverse-max-turns', type=int, default=60, help='max reverse turns while waiting for goal to appear')
    parser.add_argument('--back-step-action', type=str, default='back_fast', help='action used when goal found but ball lost')
    parser.add_argument('--back-step-times', type=int, default=1, help='times for back-step action')
    parser.add_argument('--pre-kick-forward-action', type=str, default='go_forward_one_step', help='small forward action before kicking')
    parser.add_argument('--pre-kick-forward-times', type=int, default=1, help='small forward times before kicking')
    return parser.parse_args()


def extract_ball_rgb(frame, xyxy, roi_scale: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = max(2, x2 - x1)
    bh = max(2, y2 - y1)
    half_w = max(1, int(bw * roi_scale * 0.5))
    half_h = max(1, int(bh * roi_scale * 0.5))
    rx1 = max(0, cx - half_w)
    ry1 = max(0, cy - half_h)
    rx2 = min(w, cx + half_w)
    ry2 = min(h, cy + half_h)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    bgr = np.median(roi.reshape(-1, 3), axis=0)
    b, g, r = [int(np.clip(v, 0, 255)) for v in bgr.tolist()]
    return (r, g, b), (rx1, ry1, rx2, ry2)


def pick_detection(result, target_name: str) -> Optional[Detection]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None
    names = result.names
    best = None
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        name = str(names.get(cls_id, cls_id))
        conf = float(box.conf[0].item())
        if name != target_name:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        candidate = Detection(
            name=name,
            conf=conf,
            cx=(x1 + x2) / 2.0,
            cy=(y1 + y2) / 2.0,
            xyxy=(x1, y1, x2, y2),
        )
        if best is None or candidate.conf > best.conf:
            best = candidate
    return best


def draw_detection(frame, det: Optional[Detection], color, title: str):
    if det is None:
        return
    x1, y1, x2, y2 = det.xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (int(det.cx), int(det.cy)), 4, color, -1)
    cv2.putText(frame, f'{title}:{det.conf:.2f}', (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _largest_contour_box(mask, min_area: float):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area <= best_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        best = (x, y, x + w, y + h)
        best_area = area
    return best


def detect_by_color_fallback(frame, ball_name: str, goal_name: str):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    img_area = float(max(1, h * w))
    ball = None
    goal = None

    red_mask1 = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 70, 60), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_mask = cv2.medianBlur(red_mask, 5)
    red_box = _largest_contour_box(red_mask, min_area=img_area * 0.0008)
    if red_box is not None:
        x1, y1, x2, y2 = red_box
        ball = Detection(name=ball_name, conf=0.60, cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0, xyxy=(x1, y1, x2, y2))

    blue_mask = cv2.inRange(hsv, (90, 60, 40), (135, 255, 255))
    blue_mask = cv2.medianBlur(blue_mask, 5)
    blue_box = _largest_contour_box(blue_mask, min_area=img_area * 0.002)
    if blue_box is not None:
        x1, y1, x2, y2 = blue_box
        goal = Detection(name=goal_name, conf=0.55, cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0, xyxy=(x1, y1, x2, y2))

    return ball, goal


def run_pre_approach_sequence(
    pre_ball_err: float,
    pre_ball_cy0: float,
    frame_h: int,
    args: argparse.Namespace,
    motion: MotionAdapter,
    ctx: SharedContext,
):
    pre_center_tol = max(1.0, float(args.pre_center_tol))

    # If initial ball height is already > 2h/5, skip pre-action entirely.
    if float(pre_ball_cy0) > (float(frame_h) * 0.4):
        with ctx.lock:
            ctx.dynamic_round = 0
            ctx.pre_side_move_dir = 'none'
            ctx.status_text = (
                f'pre-approach skipped: initial ball_cy0={float(pre_ball_cy0):.1f} > 2h/5={float(frame_h)*0.4:.1f}'
            )
        print(
            f'[pre-approach] skipped, initial ball_cy0={float(pre_ball_cy0):.1f}, 2h/5={float(frame_h)*0.4:.1f}'
        )
        return

    # Pre-action: pick first move direction once, keep moving until crossing to
    # the opposite side beyond tolerance, then append one more same-direction move and stop.
    side_moves = 0
    first_side_dir = 'none'
    overshoot_triggered = False
    max_side_moves = 40

    with ctx.lock:
        cur_ball = ctx.latest_ball
        cur_center_x = ctx.center_x
    if cur_ball is not None:
        cur_err = float(cur_ball.cx - cur_center_x)
        # If ball is already close to centerline (within half tolerance),
        # execute a short forward burst and skip side-move pre-action.
        if abs(cur_err) <= (pre_center_tol * 0.7):
            for _ in range(6):
                motion.run_action('go_forward_fast', times=1)
                time.sleep(0.01)
            with ctx.lock:
                ctx.dynamic_round = 0
                ctx.pre_side_move_dir = 'none'
                ctx.exit_requested = True
                ctx.phase = 'done'
                ctx.kick_done = True
                ctx.status_text = (
                    f'pre-approach: centered(|err|={abs(cur_err):.1f}<=tol*0.7={pre_center_tol*0.7:.1f}), '
                    f'forward go_forward_fast x6, program exit requested'
                )
            print(
                f'[pre-approach] centered, execute go_forward_fast x6; '
                f'|err|={abs(cur_err):.1f}, tol*0.7={pre_center_tol*0.7:.1f}; exit requested'
            )
            return
        if abs(cur_err) > pre_center_tol:
            first_side_dir = 'right' if cur_err > 0 else 'left'

    while side_moves < max_side_moves and first_side_dir != 'none':
        if first_side_dir == 'right':
            motion.run_action(args.right_move_action, times=max(1, int(args.move_times)))
        else:
            motion.run_action(args.left_move_action, times=max(1, int(args.move_times)))
        side_moves += 1
        time.sleep(0.01)

        with ctx.lock:
            cur_ball = ctx.latest_ball
            cur_center_x = ctx.center_x
        if cur_ball is None:
            break

        cur_err = float(cur_ball.cx - cur_center_x)
        if (first_side_dir == 'right' and cur_err < -pre_center_tol) or (
            first_side_dir == 'left' and cur_err > pre_center_tol
        ):
            overshoot_triggered = True
            if first_side_dir == 'right':
                motion.run_action(args.right_move_action, times=max(1, int(args.move_times)))
            else:
                motion.run_action(args.left_move_action, times=max(1, int(args.move_times)))
            side_moves += 1
            time.sleep(0.01)
            break

    with ctx.lock:
        ctx.dynamic_round = 0
        ctx.pre_side_move_dir = first_side_dir
        ctx.status_text = (
            f'pre-approach: side-move align done, initial_err={pre_ball_err:.1f}, '
            f'moves={side_moves}, tol={pre_center_tol}, first_dir={first_side_dir}, '
            f'overshoot={overshoot_triggered}, round=0'
        )
    print(
        f'[pre-approach] side-move align done: '
        f'initial_err={pre_ball_err:.1f}, moves={side_moves}, tol={pre_center_tol}'
    )

def action_worker(ctx: SharedContext, args: argparse.Namespace, motion: MotionAdapter, stop_event: threading.Event):
    # Wait until approach stage starts and then finishes.
    while not stop_event.is_set():
        with ctx.lock:
            started = ctx.approach_started
            finished = ctx.approach_done
            if not started:
                ctx.status_text = 'action thread: waiting approach stage start'
            elif not finished:
                ctx.status_text = 'action thread: waiting approach stage finish'
        if finished:
            break
        time.sleep(0.05)

    if stop_event.is_set():
        return

    with ctx.lock:
        ctx.phase = 'align_and_shoot'
        ret = ctx.approach_returncode if ctx.approach_returncode is not None else 0
        ctx.status_text = f'approach ended (code={ret}), start align+shoot'
        ctx.vision_ready_after_approach = False
    print(f'[worker] approach stage ended: returncode={ret}')

    # Wait until main thread restores camera and updates detections.
    while not stop_event.is_set():
        with ctx.lock:
            ready = ctx.vision_ready_after_approach
            if not ready:
                ctx.status_text = 'waiting camera restore after approachball...'
        if ready:
            break
        time.sleep(0.05)

    if stop_event.is_set():
        return

    state = 'line_intercept'
    align_start_ts = time.time()
    has_entered_kick = False
    kick_intersection_stable_count = 0
    last_action_t = 0.0
    action_cd = max(0.05, float(args.action_cooldown))
    def run_turn(action_name: str, times: int, status: str):
        real_times = max(1, int(times))
        motion.run_action(action_name, times=real_times)
        with ctx.lock:
            ctx.status_text = status

    while not stop_event.is_set():
        with ctx.lock:
            ball = ctx.latest_ball
            goal = ctx.latest_goal
            center_x = ctx.center_x
            dynamic_round = ctx.dynamic_round
            pre_side_move_dir = ctx.pre_side_move_dir
            program_start_ts = ctx.program_start_ts

        now = time.time()
        if (not has_entered_kick) and ((now - align_start_ts) >= 30.0):
            motion.run_action('go_forward_fast', times=5)
            with ctx.lock:
                ctx.phase = 'done'
                ctx.kick_done = True
                ctx.status_text = 'align timeout 30s without entering kick: stop align and go_forward_fast x5'
            print('[worker] align timeout 30s (not entered kick), stop align and go_forward_fast x5')
            return

        if program_start_ts is not None and (now - float(program_start_ts)) >= 50.0:
            motion.run_action('go_forward_fast', times=5)
            with ctx.lock:
                ctx.phase = 'done'
                ctx.kick_done = True
                ctx.status_text = 'align timeout 50s: stop align and go_forward_fast x5'
            print('[worker] align timeout 50s reached, stop align and go_forward_fast x5')
            return

        if (now - last_action_t) < action_cd:
            time.sleep(0.01)
            continue

        if ball is None:
            if state == 'kick_check':
                state = 'line_intercept'
                kick_intersection_stable_count = 0
                with ctx.lock:
                    ctx.status_text = 'kick_check: ball lost, back to line_intercept'
                time.sleep(0.02)
                continue

            if state == 'align_goal':
                if goal is None:
                    tilt_head_up_step(ctx, step=20, duration_ms=120)
                    if pre_side_move_dir == 'right':
                        run_turn(
                            args.turn_left_action,
                            1,
                            'align_goal: ball lost, pre-action right -> turn_left x1 to search goal',
                        )
                    else:
                        run_turn(
                            args.turn_right_action,
                            1,
                            'align_goal: ball lost, pre-action not-right -> turn_right x1 to search goal',
                        )
                    last_action_t = now
                    continue
                with ctx.lock:
                    ctx.status_text = 'align_goal: ball lost but goal visible, back step'
                motion.run_action(args.back_step_action, times=max(1, args.back_step_times))
                last_action_t = now
                continue

            # If ball is lost during alignment, turn opposite to pre-action initial turn.
            if dynamic_round < 0:
                run_turn(
                    args.turn_right_action,
                    2,
                    f'align: ball lost, pre-turn left so search right x2 (round={dynamic_round})',
                )
            elif dynamic_round > 0:
                run_turn(
                    args.turn_left_action,
                    2,
                    f'align: ball lost, pre-turn right so search left x2 (round={dynamic_round})',
                )
            else:
                motion.run_action(args.back_step_action, times=max(1, args.back_step_times))
                with ctx.lock:
                    ctx.status_text = 'align: ball lost at round=0, back step'
                last_action_t = now
                continue
            last_action_t = now
            continue

        h_ref = float(max(1, int(getattr(args, 'height', 480))))
        # During active adjustment (not kick-check / kick), keep ball height in [0.6h, 0.95h].
        if state in ('line_intercept', 'align_goal'):
            if float(ball.cy) > (0.95 * h_ref):
                motion.run_action(args.back_step_action, times=max(1, args.back_step_times))
                with ctx.lock:
                    ctx.status_text = f'pre-kick align: ball too near (cy={ball.cy:.1f}), back step'
                last_action_t = now
                print("go back!!!!!!!")
                continue
            if float(ball.cy) < (0.6 * h_ref):
                motion.run_action('go_forward_one_step', times=max(1, int(args.pre_kick_forward_times)))
                with ctx.lock:
                    ctx.status_text = f'pre-kick align: ball too far (cy={ball.cy:.1f}), forward'
                last_action_t = now
                print("go forward!!!!!!!")
                continue

        if goal is None:
            tilt_head_up_step(ctx, step=20, duration_ms=120)
            if state == 'kick_check':
                state = 'line_intercept'
                kick_intersection_stable_count = 0
                with ctx.lock:
                    ctx.status_text = 'kick_check: goal lost, back to line_intercept'
                time.sleep(0.02)
                continue

            if pre_side_move_dir == 'right':
                run_turn(
                    args.turn_left_action,
                    1,
                    'align: goal lost, pre-action right -> turn_left x1',
                )
                last_action_t = now
                continue
            else:
                run_turn(
                    args.turn_right_action,
                    1,
                    'align: goal lost, pre-action not-right -> turn_right x1',
                )
                last_action_t = now
                continue

        ball_err = ball.cx - center_x
        goal_err = goal.cx - center_x
        goal_tol = float(args.goal_face_tol) 
        # slope of line(goal-ball): large means near-vertical, intersection becomes unstable.
        dx_bg = float(goal.cx - ball.cx)
        dy_bg = float(goal.cy - ball.cy)
        if abs(dx_bg) < 1e-6:
            slope_abs = float('inf')
        else:
            slope_abs = abs(dy_bg / dx_bg)

        # Shared intersection computation (used by all align stages).
        x_intersect = center_x
        if slope_abs > 1000.0:
            x_intersect = float(ball.cx)
        else:
            dy = float(goal.cy - ball.cy)
            if abs(dy) > 1e-6:
                t = (h_ref - float(ball.cy)) / dy
                x_intersect = float(ball.cx) + t * float(goal.cx - ball.cx)
            else:
                x_intersect = 0.5 * (float(ball.cx) + float(goal.cx))
        inter_err = float(x_intersect - center_x)

        # Side relation w.r.t. center line.
        side_eps = max(5.0, float(args.center_align_tol) * 0.25)
        ball_side = 1 if ball_err > side_eps else (-1 if ball_err < -side_eps else 0)
        goal_side = 1 if goal_err > side_eps else (-1 if goal_err < -side_eps else 0)
        same_side = (ball_side != 0 and goal_side != 0 and ball_side == goal_side)

        # Stage 1: side-move based on intersection of (ball-goal) line with y=image_height.
        if state == 'line_intercept':
            # Rule A: if ball and goal are on the same side of centerline, side-move to that side.
            if same_side:
                if ball_side > 0:
                    run_turn(
                        args.turn_right_action,
                        1,
                        f'line_intercept: same-side right, turn_right x1; ball_err={ball_err:.1f}, goal_err={goal_err:.1f}',
                    )
                    with ctx.lock:
                        ctx.status_text = (
                            f'line_intercept: same-side right, turn_right; '
                            f'ball_err={ball_err:.1f}, goal_err={goal_err:.1f}'
                        )
                else:
                    run_turn(
                        args.turn_left_action,
                        1,
                        f'line_intercept: same-side left, turn_left x1; ball_err={ball_err:.1f}, goal_err={goal_err:.1f}',
                    )
                    with ctx.lock:
                        ctx.status_text = (
                            f'line_intercept: same-side left, turn_left; '
                            f'ball_err={ball_err:.1f}, goal_err={goal_err:.1f}'
                        )
                last_action_t = now
                continue

            # Rule B: only when on opposite sides, use intersection alignment.
            if abs(inter_err) > args.center_align_tol:
                if inter_err > 0:
                    motion.run_action(args.right_move_action, times=max(1, args.move_times))
                    with ctx.lock:
                        ctx.status_text = f'line_intercept: right_move, x_int={x_intersect:.1f}, err={inter_err:.1f}'
                else:
                    motion.run_action(args.left_move_action, times=max(1, args.move_times))
                    with ctx.lock:
                        ctx.status_text = f'line_intercept: left_move, x_int={x_intersect:.1f}, err={inter_err:.1f}'
                #time.sleep(0.5)
                last_action_t = now
                continue

            # Intersection aligned: switch to explicit goal-facing stage.
            state = 'align_goal'
            kick_intersection_stable_count = 0
            with ctx.lock:
                ctx.status_text = 'line_intercept done, switch to align_goal'
            continue

        if state == 'align_goal':
            # Entry condition for align_goal: intersection must remain aligned and not same-side.
            if same_side or abs(inter_err) > args.center_align_tol:
                state = 'line_intercept'
                with ctx.lock:
                    ctx.status_text = (
                        f'align_goal invalid -> line_intercept (same_side={same_side}, inter_err={inter_err:.1f})'
                    )
                continue

            if abs(goal_err) <= goal_tol:
                state = 'kick_check'
                kick_intersection_stable_count = 0
                with ctx.lock:
                    ctx.status_text = 'align_goal done, start kick_check (need 20 stable frames)'
                continue

            if abs(goal_err) > goal_tol:
                if goal_err < 0:
                    run_turn(
                        args.turn_left_action,
                        1,
                        f'align_goal: turn_left x1, goal_err={goal_err:.1f}, tol={goal_tol:.1f}',
                    )
                else:
                    run_turn(
                        args.turn_right_action,
                        1,
                        f'align_goal: turn_right x1, goal_err={goal_err:.1f}, tol={goal_tol:.1f}',
                    )
                last_action_t = now
                continue

        if state == 'kick_check':
            # Stay still and verify intersection remains within tolerance for 20 frames.
            # If entry conditions for kick_check are broken, roll back to previous step (align_goal).
            if abs(goal_err) > goal_tol or same_side:
                state = 'align_goal'
                kick_intersection_stable_count = 0
                with ctx.lock:
                    ctx.status_text = (
                        f'kick_check invalid -> align_goal (goal_err={goal_err:.1f}, same_side={same_side})'
                    )
                continue

            if abs(inter_err) <= args.center_align_tol:
                kick_intersection_stable_count += 1
                with ctx.lock:
                    ctx.status_text = (
                        f'kick_check: stable {kick_intersection_stable_count}/20, '
                        f'x_int={x_intersect:.1f}, err={inter_err:.1f}'
                    )
                if kick_intersection_stable_count >= 20:
                    state = 'kick'
                    has_entered_kick = True
                    with ctx.lock:
                        ctx.status_text = 'kick_check passed, ready to kick'
                time.sleep(0.01)
                continue

            # Any out-of-tolerance frame fails check and falls back one step.
            state = 'align_goal'
            kick_intersection_stable_count = 0
            with ctx.lock:
                ctx.status_text = (
                    f'kick_check failed (x_int err={inter_err:.1f}), back to align_goal'
                )
            continue

        if state == 'kick':
            # In kick state, keep ball height in [0.95h, h] before shooting.
            ball_top_y = float(min(int(ball.xyxy[1]), int(ball.xyxy[3])))
            if ball_top_y < (0.94 * h_ref):
                motion.run_action(args.pre_kick_forward_action, times=max(1, int(args.pre_kick_forward_times)))
                with ctx.lock:
                    ctx.status_text = f'kick-adjust: ball too far (top_y={ball_top_y:.1f}), forward'
                last_action_t = now
                print("go forward before kick!!!!!!!")
                continue
            if ball_top_y > (0.995 * h_ref):
                motion.run_action(args.back_step_action, times=max(1, args.back_step_times))
                with ctx.lock:
                    ctx.status_text = f'kick-adjust: ball too near (top_y={ball_top_y:.1f}), back'
                last_action_t = now
                print("go back before kick!!!!!!!")
                continue

            motion.stand()
            motion.run_action(args.left_shot_action, times=1)
            motion.stand()
            motion.run_action(args.right_shot_action, times=1)
            motion.stand()
            
            with ctx.lock:
                ctx.phase = 'done'
                ctx.status_text = f'done: kick actions={args.left_shot_action}+{args.right_shot_action}'
                ctx.kick_done = True
            print(f'[worker] kick actions executed: {args.left_shot_action}, {args.right_shot_action}')
            return

    with ctx.lock:
        if ctx.phase != 'done':
            ctx.status_text = 'action worker stopped'


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    bootstrap_paths(repo_root)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f'[error] ultralytics import failed: {exc}')
        return 1

    model_path = resolve_default_model(repo_root, args.model)
    if not model_path.exists():
        print(f'[error] model not found: {model_path}')
        return 1
    model = YOLO(str(model_path))
    print(f'[config] model={model_path}')
    effective_run_on_robot = (args.camera == 'robot' and not args.force_dry_run_actions) or args.run_on_robot
    print(f'[config] run_on_robot={effective_run_on_robot} (camera={args.camera}, force_dry_run_actions={args.force_dry_run_actions})')
    print('[config] round is auto-computed before approachball (right positive, left negative)')

    approach_script = Path(args.approach_script)
    if not approach_script.is_absolute():
        approach_script = repo_root / approach_script
    if not approach_script.exists():
        print(f'[error] approach script not found: {approach_script}')
        return 1
    try:
        approach_module = load_approach_module(approach_script)
    except Exception as exc:
        print(f'[error] failed to load approach module: {exc}')
        return 1
    approach_session = ApproachBallSession(approach_module)

    camera = CameraAdapter(args.camera)
    motion = MotionAdapter(run_on_robot=effective_run_on_robot)
    ctx = SharedContext()
    with ctx.lock:
        ctx.program_start_ts = time.time()
    # Set head pose early, before stable ball-color acquisition.
    set_head_pose_for_pre_approach(duration_ms=300)
    stop_event = threading.Event()
    worker = threading.Thread(target=action_worker, args=(ctx, args, motion, stop_event), daemon=True)
    worker.start()

    rgb_samples = deque(maxlen=max(1, int(args.stable_frames)))
    pre_approach_shift_done = False
    pre_approach_thread = None
    pre_approach_input_err = 0.0
    pre_approach_input_cy0 = 0.0
    pending_rgb_text = None

    try:
        while True:
            with ctx.lock:
                phase = ctx.phase
                status_text = ctx.status_text
                rgb_text = ctx.ball_rgb_text
                done = ctx.kick_done
                pre_side_move_dir = ctx.pre_side_move_dir
                exit_requested = ctx.exit_requested

            if exit_requested:
                print('[exit] pre-approach requested program exit.')
                break

            if camera is None:
                camera = CameraAdapter(args.camera)
                with ctx.lock:
                    ctx.vision_ready_after_approach = False

            # In merged mode, approachball runs in-process and consumes the same camera frames.
            if phase == 'approach_running':
                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                approach_view, approach_done = approach_session.step(frame)
                if approach_done:
                    with ctx.lock:
                        ctx.approach_done = True
                        if ctx.approach_returncode is None:
                            ctx.approach_returncode = 0
                        ctx.status_text = 'approachball finished, switching to align+shoot...'
                frame = approach_view
                cv2.putText(frame, f'phase: {phase}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, status_text[:70], (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                if rgb_text:
                    cv2.putText(frame, f'locked rgb: {rgb_text}', (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                if args.show:
                    cv2.imshow('yolo_ball_kick_dual_thread', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                continue

            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            if pre_approach_shift_done:
                if pre_side_move_dir == 'left':
                    center_x = (w // 2) + 40
                elif pre_side_move_dir == 'right':
                    center_x = (w // 2) - 40
                else:
                    center_x = (w // 2)
            else:
                center_x = (w // 2)
            kwargs = {'source': frame, 'conf': float(args.conf), 'imgsz': int(args.imgsz), 'verbose': False}
            if args.device:
                kwargs['device'] = args.device
            results = model.predict(**kwargs)
            result = results[0] if results else None

            ball = pick_detection(result, args.ball_class)
            goal = pick_detection(result, args.goal_class)
            if not args.disable_color_fallback and (ball is None or goal is None):
                fb_ball, fb_goal = detect_by_color_fallback(frame, args.ball_class, args.goal_class)
                if ball is None:
                    ball = fb_ball
                if goal is None:
                    goal = fb_goal

            with ctx.lock:
                ctx.latest_ball = ball
                ctx.latest_goal = goal
                ctx.center_x = center_x
                if ctx.phase in ('align_and_shoot', 'done'):
                    ctx.vision_ready_after_approach = True

            draw_detection(frame, ball, (0, 0, 255), 'ball')
            draw_detection(frame, goal, (0, 255, 0), 'goal')
            cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 0), 1)

            # Lock ball color first, then launch approachball in-process.
            with ctx.lock:
                proc_started = ctx.approach_started
            if (not proc_started) and (ball is not None):
                extracted = extract_ball_rgb(frame, ball.xyxy, float(args.roi_scale))
                if extracted is not None:
                    rgb, roi_box = extracted
                    rgb_samples.append(rgb)
                    rx1, ry1, rx2, ry2 = roi_box
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
                    cv2.putText(frame, f'ball rgb={rgb}', (rx1, max(16, ry1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

                    if len(rgb_samples) >= rgb_samples.maxlen:
                        if pending_rgb_text is None:
                            avg = np.array(rgb_samples, dtype=np.float32).mean(axis=0)
                            r, g, b = [int(np.clip(v, 0, 255)) for v in avg.tolist()]
                            pending_rgb_text = f'{r},{g},{b}'
                        rgb_text = pending_rgb_text
                        pre_actions_ready = True
                        # Before approachball: run pre-actions in a dedicated thread.
                        if not pre_approach_shift_done:
                            if pre_approach_thread is None:
                                pre_approach_input_err = float(ball.cx - center_x)
                                pre_approach_input_cy0 = float(ball.cy)
                                pre_approach_thread = threading.Thread(
                                    target=run_pre_approach_sequence,
                                    args=(pre_approach_input_err, pre_approach_input_cy0, h, args, motion, ctx),
                                    daemon=True,
                                )
                                with ctx.lock:
                                    ctx.status_text = (
                                        f'pre-approach thread started, initial ball_err={pre_approach_input_err:.1f}, '
                                        f'ball_cy0={pre_approach_input_cy0:.1f}'
                                    )
                                pre_approach_thread.start()

                            if pre_approach_thread is not None and pre_approach_thread.is_alive():
                                with ctx.lock:
                                    ctx.status_text = 'pre-approach thread running...'
                                pre_actions_ready = False

                            if pre_approach_thread is not None and not pre_approach_thread.is_alive():
                                pre_approach_shift_done = True
                                pre_approach_thread = None
                                pre_actions_ready = True
                        if not pre_actions_ready:
                            # Keep showing live image while pre-action thread is running.
                            with ctx.lock:
                                ctx.ball_rgb_text = rgb_text
                        else:
                            with ctx.lock:
                                ctx.ball_rgb_text = rgb_text

            # Launch approach stage as soon as pre-actions are done.
            if (not proc_started) and pre_approach_shift_done and (pending_rgb_text is not None):
                with ctx.lock:
                    current_round = int(ctx.dynamic_round)
                print(f'[launch] detected stable ball rgb={pending_rgb_text}')
                print(f'[launch] merged approach mode: call approachball module functions directly')

                if args.dry_run_subprocess:
                    with ctx.lock:
                        ctx.approach_returncode = 0
                        ctx.approach_started = True
                        ctx.approach_done = True
                        ctx.phase = 'align_and_shoot'
                        ctx.status_text = 'dry-run approach, directly align+shoot'
                else:
                    try:
                        approach_session.start(
                            target_rgb_text=pending_rgb_text,
                            align_round=current_round,
                            debug=False,
                        )
                        with ctx.lock:
                            ctx.ball_rgb_text = pending_rgb_text
                            ctx.approach_started = True
                            ctx.approach_done = False
                            ctx.phase = 'approach_running'
                            ctx.status_text = 'approachball running in-process...'
                    except Exception as exc:
                        with ctx.lock:
                            ctx.approach_started = True
                            ctx.approach_done = True
                            ctx.approach_returncode = 1
                            ctx.phase = 'done'
                            ctx.kick_done = True
                            ctx.status_text = f'approach start failed: {exc}'
                        print(f'[error] approach start failed: {exc}')

            with ctx.lock:
                phase = ctx.phase
                status_text = ctx.status_text
                rgb_text = ctx.ball_rgb_text
                done = ctx.kick_done

            cv2.putText(frame, f'phase: {phase}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, status_text[:70], (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            if rgb_text:
                cv2.putText(frame, f'locked rgb: {rgb_text}', (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            if args.show:
                cv2.imshow('yolo_ball_kick_dual_thread', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break

            if done:
                print('[done] align-and-shoot completed')
                break
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        approach_session.abort()
        if camera is not None:
            camera.close()
        if args.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

