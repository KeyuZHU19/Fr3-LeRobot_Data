# Oculus Quest 遥操作快速指南

## 控制器按键

| 按键 | 功能 |
|------|------|
| **RG（右手握持）** | 按住才能移动机器人 |
| **RTr（右手扳机）** | 按下关闭夹爪，松开打开夹爪 |
| **A 按钮** | 复位机器人到初始位置 |
| **右手控制器位姿** | 控制末端执行器位置和姿态 |

录制控制键：

| 按键 | 功能 |
|------|------|
| **右方向键** | 保存当前 episode，进入复位阶段 |
| **左方向键** | 丢弃当前 episode，重新录制 |
| **Esc** | 结束所有录制并保存 |

---

## 启动流程

### 第一步：NUC 端（polymetis-local 环境）

依次启动三个服务：

```bash
# 终端 1：启动 robot server
conda activate polymetis-local
launch_robot.py robot_client=franka_hardware

# 终端 2：启动 gripper server
conda activate polymetis-local
launch_gripper.py gripper=franka_hand

# 终端 3：启动 interface server
conda activate polymetis-local
python franka_interface_server.py
```

### 第二步：确认 Oculus 连接

```bash
adb devices   # 确认 Quest 已出现在列表中
```

若无设备，重启 ADB：
```bash
adb kill-server && adb start-server
```

### 第三步：配置 `scripts/config/record_cfg.yaml`

```yaml
record:
  repo_id: "你的HF用户名/任务描述"   # 例如 babycare/pick_apple
  control_mode: "oculus"

  teleop:
    oculus_config:
      ip: null           # USB 连接填 null；无线填 Quest IP，例如 "192.168.1.62"
      pose_scaler: [2.0, 1.5]   # [位置缩放, 姿态缩放]

  robot:
    ip: "127.0.0.1"      # franka_interface_server 所在机器 IP

  storage:
    push_to_hub: false   # 录完自动上传 HuggingFace 改为 true
```

### 第四步：开始录制

```bash
conda activate franka_data
franka-record
```

---

## 参数调整

### 末端姿态不够灵活（旋转响应慢）

调大 `pose_scaler` 第二个值：

```yaml
pose_scaler: [2.0, 2.5]  # 姿态缩放 1.5 → 2.5
```

Quest 控制器转动相同角度，机器人末端旋转幅度更大。

### 手臂够不到某些位置

调大 `pose_scaler` 第一个值：

```yaml
pose_scaler: [3.5, 1.5]  # 位置缩放 2.0 → 3.5
```

Quest 移动更小的距离，机器人移动更大的距离，相当于"放大"手臂长度。

> **注意**：两个值不能调太大。位置缩放过大会导致速度积分跳变过快，容易触发急停；姿态缩放过大会导致末端旋转过猛。建议每次小幅调整（+0.5）后测试。

---

## 常见问题

**机器人无法移动**
- 确认按住 RG（右手握持键）
- 检查 `record_cfg.yaml` 中 `placo.robot_ip` 是否与 franka_interface_server 在同一机器（本机填 `127.0.0.1`）

**触发急停（User Stop）**
- 检查是否碰撞或超出工作空间
- 在 Franka Desk 释放急停，点击解锁制动器，重新切换 Remote Control 模式，重启 franka_interface_server

**夹爪无响应**
- 确认 gripper server 已启动
- 按 A 复位后重试

**ADB 连接失败**
```bash
adb kill-server && adb start-server && adb devices
```
