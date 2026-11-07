# Live rear-radar detector (ROS 2 node)

A ROS 2 node that runs the trained RadarPillar car detector on the rear FRGen21 radar, live
or from a bag. It runs on a laptop CPU and needs no GPU, OpenPCDet or dataset.

```text
/radar_rear/frgen21/pointcloud ─┐                         ┌─▶ /rear_radar/detections  (vision_msgs/Detection3DArray)
                                ├─▶ rear_radar_node.py ───┼─▶ /rear_radar/markers     (MarkerArray)
/vehicle/uva_odometry ──────────┘                         ├─▶ /rear_radar/points      (model input, current scan)
                                                          └─▶ /rear_radar/status      (JSON: cars, latency, speed)
```

Outputs are in `rear_axle_middle`, stamped with the scan's header time. A box is centre
x, y, z, then l, w, h and yaw.

| file | what it is |
|---|---|
| `rear_radar_node.py` | the node |
| `rear_radar_model.py` | the network and post-processing in plain PyTorch |
| `models/rear_radarpillar_v5.pt` | exported weights, 0.7 MB (v5 seed 1, epoch 15). Not in git (`*.pt`) |
| `export_model.py` | lab only: writes the `.pt` from a checkpoint and checks it against OpenPCDet |

## Setup

Written for **Kilted** (Ubuntu 24.04), tested 2026-11-03; it also runs on Humble. `vision_msgs`
is optional; without it the node skips `/rear_radar/detections`.

```bash
sudo apt install ros-kilted-vision-msgs ros-kilted-rosbag2-storage-mcap python3-venv
```

Copy the code and model onto the laptop. The node imports `scan_features` from the dataset
builder, so three top-level files come too:

```bash
mkdir -p ~/rear_radar && rsync -av <host>:<path-to-this-repo>/{build_nuscenes_dataset.py,repo_config.py,av24_geometry.py,live} ~/rear_radar/ --exclude __pycache__
```

Ubuntu 24.04 won't let pip install into the system Python, so make a venv that still sees
`rclpy` and the ROS messages. Source ROS first:

```bash
source /opt/ros/kilted/setup.bash && python3 -m venv --system-site-packages ~/rear_radar/.venv && source ~/rear_radar/.venv/bin/activate
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r ~/rear_radar/live/requirements.txt
```

## Run

In each terminal, `source /opt/ros/kilted/setup.bash && source ~/rear_radar/.venv/bin/activate` first.

```bash
cd ~/rear_radar && python3 live/rear_radar_node.py --ros-args -p use_sim_time:=true
```

```bash
ros2 bag play live/demo_bags/laguna_0726_run2_a --clock
```

Leave out `use_sim_time` and the bag on live data. Each scan takes about 40–55 ms on a
4-thread CPU; the radar runs at 16 Hz.

| parameter | default | |
|---|---|---|
| `radar_topic` | `/radar_rear/frgen21/pointcloud` | needs fields x y z velocity rcs confidence snr power |
| `odom_topic` | `/vehicle/uva_odometry` | twist in `center_of_gravity`, for Doppler compensation |
| `score_threshold` | 0.3 | minimum score for `detections` |
| `gate_min_speed` | true | publish no cars below 5 m/s ego speed (training never saw that) |
| `twist_wait_s` | 0.1 | how long a scan waits for odometry past its own stamp |
| `device` | auto | `cpu` or `cuda` |
| `model` | `live/models/rear_radarpillar_v5.pt` | |

## Demo bags

`live/demo_bags/` (not in git) holds two windows of **Laguna Seca 2026-07-26 run2**, a
session no model trained on. They carry only the radar, odometry and `/tf`.

- `laguna_0726_run2_a` (220 s): a car follows 31–41 m behind most of the time. At ~95–115 s
  the ego odometry reads 0 m/s, so those scans are gated.
- `laguna_0726_run2_b` (180 s): the car drops back to ~70 m, closes again, then ~70 s with
  no car behind.

Run2 has no usable ground truth, so these show behaviour on unseen data, not accuracy.

## Exporting another checkpoint

On the lab machines:

```bash
PYTHONPATH=training/vendor/RadarPillar/.deps/native:training/vendor/RadarPillar python3 live/export_model.py --ckpt PATH --out live/models/NAME.pt --check 200
```

`--check` compares against OpenPCDet; v5 matched it to within 2e-5 on 400 `val_qa` samples.
Then pass `-p model:=live/models/NAME.pt`.
