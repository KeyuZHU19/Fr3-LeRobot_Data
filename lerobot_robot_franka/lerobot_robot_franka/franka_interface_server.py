'''
run on the NUC connected to the franka robot
to provide zerorpc server interface

Velocity-extrapolation control loop (50Hz) runs as a gevent greenlet:
  - Client pushes 6D EE velocity at ~15Hz via robot_set_velocity()
  - Greenlet integrates velocity × dt into _current_target at 50Hz
  - Sends _current_target to Polymetis at 50Hz via threadpool (no gevent starvation)
  - When velocity expires (RG released, max_age_s=0.15s), integration stops;
    robot holds _current_target via Cartesian impedance spring
Result: smooth continuous motion at 50Hz instead of 15Hz move-stop pattern.
'''

import socket
import struct
import threading
import time as _time
import gevent
import gevent.threadpool
import zerorpc
from polymetis import RobotInterface
from polymetis import GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)


class LatestValue:
    """Thread-safe buffer that always holds the most recent value.
    Automatically expires stale entries after max_age_s seconds.
    Design from Reference/fr3_teleop/latest.py (AV-ALOHA pattern).
    """
    def __init__(self, max_age_s: float = 0.15):
        self._value = None
        self._timestamp = 0.0
        self._lock = threading.Lock()
        self._max_age_s = max_age_s  # auto-expire stale targets

    def put(self, value):
        with self._lock:
            self._value = value
            self._timestamp = _time.monotonic()

    def get(self):
        with self._lock:
            if self._value is None:
                return None
            if _time.monotonic() - self._timestamp > self._max_age_s:
                self._value = None  # expired
                return None
            return self._value

    def clear(self):
        with self._lock:
            self._value = None


class FrankaInterfaceServer:
    def __init__(self):
        self._velocity = LatestValue(max_age_s=0.06)  # 6D velocity [m/s, rad/s]; 0.06s = 3 client ticks at 50Hz
        self._current_target = None           # np.ndarray [6]: integrated EE pose target
        self._control_greenlet = None         # gevent greenlet handle
        self._control_hz = 50                 # 50Hz: safe for gRPC roundtrip (~10ms)
        self._controller_active = False
        # ThreadPool(1): run blocking Polymetis gRPC calls without blocking gevent event loop
        self._threadpool = gevent.threadpool.ThreadPool(1)

        try:
            self.robot = RobotInterface(enforce_version=False)
            log.info("Connected to robot")
        except:
            log.error("Failed to connect to robot")

        # UDP velocity listener (port 4243): receives 48-byte packets from background thread.
        # Bypasses gevent/zerorpc threading constraints entirely.
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.bind(('0.0.0.0', 4243))
        self._udp_sock.settimeout(0.05)
        self._udp_thread = threading.Thread(target=self._udp_velocity_loop, daemon=True)
        self._udp_thread.start()
        log.info("[UDP] Velocity listener started on port 4243")

    # ──────────────────────────────────────────────
    # Gripper
    # ──────────────────────────────────────────────
    def gripper_initialize(self):
        try:
            self.gripper = GripperInterface()
            log.info("Connected to gripper")
        except:
            log.error("Failed to connect to gripper")

    def gripper_goto(self, width, speed, force,
                     epsilon_inner=-1.0, epsilon_outer=-1.0, blocking=True):
        if not hasattr(self, 'gripper'):
            return
        self.gripper.goto(width=width, speed=speed, force=force, blocking=blocking)

    def gripper_grasp(self, speed, force, grasp_width=0.0,
                      epsilon_inner=-1.0, epsilon_outer=-1.0, blocking=True):
        if not hasattr(self, 'gripper'):
            return
        self.gripper.grasp(speed=speed, force=force, grasp_width=grasp_width,
                           epsilon_inner=epsilon_inner, epsilon_outer=epsilon_outer,
                           blocking=blocking)

    def gripper_get_state(self) -> dict:
        if not hasattr(self, 'gripper'):
            return {"width": 0.0, "is_moving": False, "is_grasped": False,
                    "prev_command_successful": False, "error_code": -1}
        state = self.gripper.get_state()
        return {
            "width": state.width,
            "is_moving": state.is_moving,
            "is_grasped": state.is_grasped,
            "prev_command_successful": state.prev_command_successful,
            "error_code": state.error_code,
        }

    # ──────────────────────────────────────────────
    # Robot state queries
    # ──────────────────────────────────────────────
    def robot_get_joint_positions(self) -> list:
        return self.robot.get_joint_positions().numpy().tolist()

    def robot_get_joint_velocities(self) -> list:
        return self.robot.get_joint_velocities().numpy().tolist()

    def robot_get_ee_pose(self) -> list:
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()

    # ──────────────────────────────────────────────
    # Motion commands (blocking, for reset/homing)
    # ──────────────────────────────────────────────
    def robot_move_to_joint_positions(self, positions, time_to_go=None,
                                       delta=False, Kq=None, Kqd=None):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go,
            delta=delta,
            Kq=torch.Tensor(Kq) if Kq is not None else None,
            Kqd=torch.Tensor(Kqd) if Kqd is not None else None,
        )

    def robot_go_home(self):
        self.robot.go_home()

    def robot_move_to_ee_pose(self, pose=None, time_to_go=None, delta=False,
                               Kx=None, Kxd=None, op_space_interp=True):
        pose = torch.Tensor(pose)
        self.robot.move_to_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat()),
            time_to_go=time_to_go,
            delta=delta,
            Kx=torch.Tensor(Kx) if Kx is not None else None,
            Kxd=torch.Tensor(Kxd) if Kxd is not None else None,
            op_space_interp=op_space_interp,
        )

    # ──────────────────────────────────────────────
    # Controller management
    # ──────────────────────────────────────────────
    def robot_start_joint_impedance_control(self, Kq=None, Kqd=None, adaptive=True):
        self._stop_control_loop()
        self.robot.start_joint_impedance(
            Kq=torch.Tensor(Kq) if Kq is not None else None,
            Kqd=torch.Tensor(Kqd) if Kqd is not None else None,
            adaptive=adaptive,
        )
        self._controller_active = True

    def robot_start_cartesian_impedance_control(self, Kx=None, Kxd=None):
        self._stop_control_loop()
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx) if Kx is not None else None,
            Kxd=torch.Tensor(Kxd) if Kxd is not None else None,
        )
        self._velocity.clear()
        # Seed _current_target from actual robot EE pose so integration starts at the right place
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        self._current_target = np.concatenate([pos, rot_vec])
        self._controller_active = True
        self._start_control_loop()
        log.info(f"[CTRL] Cartesian impedance started at {self._control_hz}Hz, target seeded from EE pose: {np.round(self._current_target, 4).tolist()}")

    def robot_terminate_current_policy(self):
        self._stop_control_loop()
        self.robot.terminate_current_policy()
        self._controller_active = False

    # ──────────────────────────────────────────────
    # High-frequency control interface (velocity-based)
    # ──────────────────────────────────────────────
    def robot_set_velocity(self, velocity: list):
        """Non-blocking: store 6D EE velocity [vx,vy,vz,wx,wy,wz] in m/s and rad/s.
        The 50Hz control greenlet integrates this to produce a smooth continuous target."""
        self._velocity.put(velocity)

    def robot_clear_velocity(self):
        """Explicitly clear velocity (called when RG released).
        Integration stops; robot holds current position via impedance spring."""
        self._velocity.clear()

    def robot_update_desired_ee_pose(self, pose: list):
        """Legacy direct call (kept for compatibility)."""
        pose = torch.Tensor(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat()),
        )

    def robot_update_desired_joint_positions(self, positions: list):
        self.robot.update_desired_joint_positions(
            positions=torch.Tensor(positions)
        )

    # ──────────────────────────────────────────────
    # Internal: UDP velocity receiver (threading, not gevent)
    # ──────────────────────────────────────────────
    def _udp_velocity_loop(self):
        """Receive 6D velocity packets via UDP from OculusRobot background thread.
        Packet format: 6 IEEE-754 doubles (48 bytes), little-endian.
        Thread-safe: only writes to LatestValue which has its own lock.
        """
        _PACKET = struct.Struct('<6d')
        while True:
            try:
                data, _ = self._udp_sock.recvfrom(64)
                if len(data) == _PACKET.size:
                    velocity = list(_PACKET.unpack(data))
                    self._velocity.put(velocity)
            except socket.timeout:
                pass
            except Exception as e:
                log.warning(f"[UDP] Error: {e}")

    # ──────────────────────────────────────────────
    # Internal: gevent control greenlet
    # ──────────────────────────────────────────────
    def _start_control_loop(self):
        self._control_greenlet = gevent.spawn(self._run_control_loop)

    def _stop_control_loop(self):
        self._velocity.clear()
        if self._control_greenlet is not None:
            self._control_greenlet.kill(block=False)
            self._control_greenlet = None

    def _polymetis_update_ee_safe(self, target: list) -> str:
        """Blocking Polymetis call in threadpool. Returns status string, never raises.
        Catching exceptions here prevents gevent threadpool from printing tracebacks.
        """
        try:
            t = torch.Tensor(target)
            self.robot.update_desired_ee_pose(
                position=t[:3],
                orientation=torch.Tensor(st.Rotation.from_rotvec(t[3:].numpy()).as_quat()),
            )
            return 'ok'
        except Exception as e:
            if "no controller" in str(e).lower():
                return 'no_controller'
            return 'error'

    def _run_control_loop(self):
        """Gevent greenlet: integrates latest velocity into _current_target at ~50Hz,
        then sends target to Polymetis via threadpool (non-blocking for gevent loop).

        Velocity extrapolation pattern (Reference/fr3_teleop/mapper.py):
          client sends velocity at ~15Hz → server integrates at 50Hz
          → smooth continuous motion instead of 15Hz move-stop pattern.
        When velocity expires (RG released, max_age_s=0.30s), integration stops and
        robot holds _current_target via the impedance spring.

        Uses actual elapsed time (not hardcoded dt) so integration is accurate
        even when gRPC calls take variable time.
        """
        target_dt = 1.0 / self._control_hz
        log.info(f"[CTRL] Control greenlet started at {self._control_hz}Hz (velocity integration)")
        last_tick = _time.monotonic()
        while True:
            now = _time.monotonic()
            dt_actual = min(now - last_tick, 0.1)  # cap at 100ms to avoid large jumps on stall
            last_tick = now

            velocity = self._velocity.get()
            if velocity is not None and self._current_target is not None:
                vel = np.array(velocity)
                # Integrate using actual elapsed time, not assumed dt
                self._current_target[:3] += vel[:3] * dt_actual
                if np.linalg.norm(vel[3:]) > 1e-8:
                    delta_rot = st.Rotation.from_rotvec(vel[3:] * dt_actual)
                    current_rot = st.Rotation.from_rotvec(self._current_target[3:])
                    self._current_target[3:] = (delta_rot * current_rot).as_rotvec()
                result = self._threadpool.spawn(
                    self._polymetis_update_ee_safe, self._current_target.tolist()
                ).get()
                if result == 'no_controller':
                    log.warning("[CTRL] Workspace limit hit, controller stopped. Press A on Quest to reset.")
                    self._velocity.clear()
            gevent.sleep(target_dt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("zerorpc").setLevel(logging.ERROR)  # suppress heartbeat noise
    server = FrankaInterfaceServer()
    s = zerorpc.Server(server)
    s.bind("tcp://0.0.0.0:4242")
    s.run()
