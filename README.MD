# Franka Quest Teleoperation Setup

This README documents the **current working setup on this PC** for:

- environment installation
- Franka service startup
- Meta Quest teleoperation
- dataset recording with the current `record_cfg.yaml`

It intentionally does **not** cover the older isomorphic-arm workflow, SpaceMouse, or the historical RealSense setup.

## Scope

This repository is currently used in the following way:

- `polymetis-local` runs the Franka robot, gripper, and interface services
- `franka_data` runs dataset recording
- Meta Quest is used as the teleoperation device
- the recorder is configured for `control_mode: "oculus"`
- the robot service is reached locally at `127.0.0.1`
- the current camera setup is `3 x ZED` with:
  - `wrist_zed_serial: 35207644`
  - `exterior_zed_serial: 34533645`
  - `exterior2_zed_serial: 11130146`

## Repository Layout

- `scripts/config/record_cfg.yaml`
  Current recording configuration.
- `scripts/core/run_record.py`
  Main recording entry point behind `franka-record`.
- `lerobot_robot_franka/`
  Franka robot wrapper and interface client/server bridge.
- `lerobot_teleoperator_franka/`
  Teleoperator implementations, including Oculus/Quest support.

## 1. Prerequisites

This README assumes:

- Ubuntu Linux
- Conda installed
- a working local `polymetis-local` environment for Franka
- Franka FCI is enabled on Desk
- Meta Quest developer mode is enabled
- `adb` is available on the machine

## 2. Python Environments

Two Conda environments are used in the current setup.

### 2.1 `polymetis-local`

This environment is used to run:

- `launch_robot.py`
- `launch_gripper.py`
- `franka_interface_server.py`

This repo does **not** recreate Polymetis from scratch. It assumes your local Franka/Polymetis stack is already installed and working.

### 2.2 `franka_data`

This environment is used to run:

- `franka-record`
- dataset utilities from this repository

## 3. Install the Recording Environment

### 3.1 Create the environment

```bash
conda create -n franka_data python=3.10 -y
conda activate franka_data
```

### 3.2 Install LeRobot

This project expects a local editable LeRobot checkout. The setup that has been used on this machine is:

```bash
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
pip install -e .
```

### 3.3 Install this repository

```bash
cd ~/lerobot_franka_teleop
pip install -e .
```

This installs the console commands from [setup.py](/home/franka/lerobot_franka_teleop/setup.py:23), including:

- `franka-record`
- `franka-replay`
- `franka-visualize`
- `franka-reset`
- `franka-help`

### 3.4 Install ADB

```bash
sudo apt install android-tools-adb
adb version
```

## 4. Oculus / Quest Setup

This project uses the vendored `oculus_reader` inside:

- `lerobot_teleoperator_franka/lerobot_teleoperator_franka/oculus/oculus_reader`

The APK is already included in the repo.

### 4.1 Authorize the headset

Connect the Quest by USB and run:

```bash
adb devices
```

You want to see:

```bash
<device_id>    device
```

If you see `unauthorized`, put on the headset and accept the USB debugging prompt.

### 4.2 Install or reinstall the teleop APK

```bash
cd ~/lerobot_franka_teleop/lerobot_teleoperator_franka/lerobot_teleoperator_franka/oculus/oculus_reader
adb install -r oculus_reader/APK/teleop-debug.apk
```

After installation, launch the app from **Unknown Sources** on the headset.

### 4.3 Current connection mode

The current `record_cfg.yaml` uses:

```yaml
teleop:
  oculus_config:
    ip: null
```

That means the current setup expects **USB ADB**, not wireless Quest ADB.

## 5. Start Franka Services

Open three terminals and run the following in `polymetis-local`.

### Terminal 1: Robot server

```bash
conda activate polymetis-local
launch_robot.py robot_client=franka_hardware
```

### Terminal 2: Gripper server

```bash
conda activate polymetis-local
launch_gripper.py gripper=franka_hand
```

### Terminal 3: Interface server

```bash
conda activate polymetis-local
cd ~/lerobot_franka_teleop/lerobot_robot_franka/lerobot_robot_franka
python franka_interface_server.py
```

Notes:

- This repo currently uses `127.0.0.1` in [record_cfg.yaml](/home/franka/lerobot_franka_teleop/scripts/config/record_cfg.yaml:60), so these services are expected to run on the same machine as `franka-record`.
- If Franka Desk is not in `Remote Control` and `FCI` mode, the robot service will fail.

## 6. Current Recording Configuration

The active configuration lives in [record_cfg.yaml](/home/franka/lerobot_franka_teleop/scripts/config/record_cfg.yaml:1).

Important current settings:

- `control_mode: "oculus"`
- `fps: 15`
- `display: False`
- `camera_type: "zed"`
- `use_videos: True`
- `execute_mode: "ee_pose"`
- Quest pose scaling:
  - `pose_scaler: [2.5, 1.5]`
- Quest channel signs:
  - `[1, 1, 1, -1, -1, 1]`
- Gripper tuning:
  - `gripper_max_open: 0.08`
  - `gripper_force: 70.0`
  - `gripper_speed: 0.10`
  - `gripper_grasp_width: 0.030`
  - `gripper_epsilon_inner: 0.006`
  - `gripper_epsilon_outer: 0.006`

## 7. Start Recording

In a new terminal:

```bash
conda activate franka_data
cd ~/lerobot_franka_teleop
franka-record
```

If the Quest is visible and the three ZED cameras are available, the startup flow should reach:

- robot connected
- gripper initialized
- cameras initialized
- `OculusTeleop connected successfully`

## 8. Quest Controls

Current Quest controls:

- `RG`
  Hold to enable robot motion.
- `RTr`
  Toggle gripper open/close on each press.
- `A`
  Request robot reset to the home position.
- Right controller pose
  Controls end-effector motion.

Current keyboard controls during recording:

- `Right Arrow`
  Finish the current episode and move to the reset stage.
- `Left Arrow`
  Drop the current episode and re-record it.
- `Esc`
  Stop recording.

Important behavior:

- pressing `Enter` at the `WAIT` prompt only enters the reset stage
- pressing `A` actually requests robot reset

## 9. Cameras

The current configuration is based on `3` ZED cameras:

- `wrist_image` -> `35207644`
- `exterior_image` -> `34533645`
- `exterior2_image` -> `11130146`

Current ZED settings:

- resolution: `672 x 376`
- nominal fps: `15`
- `exterior2_zed_rotation: 180`
- ZED depth is disabled in the current local runtime patch path

If you want to verify the cameras before recording:

```bash
ZED_Explorer -a
```

## 10. Common Checks

### 10.1 Quest is not visible

```bash
adb kill-server
adb start-server
adb devices
```

### 10.2 ZED cameras are missing

```bash
ZED_Explorer -a
```

If the cameras disappear intermittently, the issue is usually USB topology / power / cable stability, especially with multiple ZED devices attached at once.

### 10.3 Robot cannot start

Check:

- Franka Desk is in `Remote Control`
- FCI is enabled
- the robot is not in a stop state

### 10.4 Gripper behavior looks wrong

Check the gripper parameters in [record_cfg.yaml](/home/franka/lerobot_franka_teleop/scripts/config/record_cfg.yaml:66), especially:

- `gripper_force`
- `gripper_grasp_width`
- `gripper_epsilon_inner`
- `gripper_epsilon_outer`

## 11. Practical Workflow

Typical workflow on this PC:

1. Connect and authorize Quest over USB.
2. Confirm `adb devices` shows `device`.
3. Confirm `ZED_Explorer -a` sees the three cameras.
4. Start the three Franka services in `polymetis-local`.
5. Start `franka-record` in `franka_data`.
6. Record with Quest.

## 12. Related File

For a shorter operator-facing cheat sheet, see [TELEOP_GUIDE.md](/home/franka/lerobot_franka_teleop/TELEOP_GUIDE.md:1).
