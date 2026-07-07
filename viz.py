#!/usr/bin/env python3
"""
Vive tracker real-time 3D visualizer.
Subscribes to /tf (trackers) and /tf_static (lighthouses) and plots live.

Usage:
    source install/setup.bash
    python3 visualize.py
"""

import threading
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

TRAIL_LEN = 300
UPDATE_HZ = 30
TRACKER_MARKER = 'o'
LIGHTHOUSE_MARKER = '^'
AXIS_LEN = 0.25
X_LIMIT = (-1.0, 1.0)
Y_LIMIT = (-1.0, 1.0)
Z_LIMIT = (0.0, 2.0)
VALUE_DECIMALS = 3


def _quat_to_rot(q):
    """(w, x, y, z) -> 3x3 rotation matrix (columns = local axes)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


def _fmt_vec(values, decimals=VALUE_DECIMALS):
    return ', '.join(f'{v:.{decimals}f}' for v in values)


class ViveVisualizer(Node):
    def __init__(self):
        super().__init__('vive_visualizer')
        self.lock = threading.Lock()
        self.trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
        self.current = {}
        self.lighthouses = {}

        self.create_subscription(TFMessage, '/tf', self._on_tf, 10)
        self.create_subscription(TFMessage, '/tf_static', self._on_tf_static, 10)

    def _on_tf(self, msg: TFMessage):
        with self.lock:
            for t in msg.transforms:
                p = t.transform.translation
                q = t.transform.rotation
                name = t.child_frame_id
                pos = (p.x, p.y, p.z)
                self.trails[name].append(pos)
                self.current[name] = (pos, (q.w, q.x, q.y, q.z))

    def _on_tf_static(self, msg: TFMessage):
        with self.lock:
            for t in msg.transforms:
                p = t.transform.translation
                self.lighthouses[t.child_frame_id] = (p.x, p.y, p.z)


def _spin(node: Node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = ViveVisualizer()

    thread = threading.Thread(target=_spin, args=(node,), daemon=True)
    thread.start()

    colors = plt.cm.tab10.colors

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title('Vive Tracker Visualizer')

    print("Waiting for tracker data... (Ctrl+C to quit)")

    try:
        while rclpy.ok():
            ax.cla()
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('Vive Tracker Live Pose')
            ax.set_xlim(*X_LIMIT)
            ax.set_ylim(*Y_LIMIT)
            ax.set_zlim(*Z_LIMIT)

            with node.lock:
                value_lines = []

                # Draw lighthouses
                for lh_name, (x, y, z) in node.lighthouses.items():
                    ax.scatter([x], [y], [z],
                               marker=LIGHTHOUSE_MARKER, s=120, c='gray',
                               zorder=6, label=f'LH:{lh_name}')
                    lh_label = f'{lh_name}\nxyz=({_fmt_vec((x, y, z))})'
                    ax.text(x, y, z + 0.05, lh_label, fontsize=7, color='gray')
                    value_lines.append(f'LH {lh_name}: xyz=({_fmt_vec((x, y, z))})')

                # Draw tracker trails + current position + orientation axes
                for i, (name, trail) in enumerate(node.trails.items()):
                    if not trail:
                        continue
                    color = colors[i % len(colors)]
                    pts = np.array(trail)
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                            '-', color=color, alpha=0.35, linewidth=1)
                    ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]],
                               marker=TRACKER_MARKER, s=80, c=[color],
                               zorder=7, label=name)

                    # Draw local orientation axes
                    if name in node.current:
                        pos, quat = node.current[name]
                        R = _quat_to_rot(quat)
                        ox, oy, oz = pos
                        pose_label = (
                            f'{name}\n'
                            f'xyz=({_fmt_vec(pos)})\n'
                            f'q(wxyz)=({_fmt_vec(quat)})'
                        )
                        ax.text(ox, oy, oz + 0.05,
                                pose_label, fontsize=8, color=color)
                        value_lines.append(
                            f'{name}: xyz=({_fmt_vec(pos)}) q(wxyz)=({_fmt_vec(quat)})'
                        )
                        for axis, col, lbl in zip(R.T,
                                                  ['red', 'lime', 'dodgerblue'],
                                                  ['x', 'y', 'z']):
                            dx, dy, dz = axis * AXIS_LEN
                            ax.quiver(ox, oy, oz, dx, dy, dz,
                                      color=col, linewidth=2.5,
                                      arrow_length_ratio=0.2,
                                      zorder=8)
                            ax.text(ox + dx * 1.1, oy + dy * 1.1, oz + dz * 1.1,
                                    lbl, color=col, fontsize=9, fontweight='bold',
                                    zorder=9)
                    else:
                        ax.text(pts[-1, 0], pts[-1, 1], pts[-1, 2] + 0.05,
                                name, fontsize=8, color=color)

                if value_lines:
                    ax.text2D(0.02, 0.98, '\n'.join(value_lines),
                              transform=ax.transAxes,
                              va='top', ha='left',
                              fontsize=8, family='monospace',
                              bbox={
                                  'boxstyle': 'round,pad=0.35',
                                  'facecolor': 'white',
                                  'edgecolor': '0.7',
                                  'alpha': 0.85,
                              })

            # Draw world coordinate axes at origin
            axis_len = 0.3
            ax.quiver(0, 0, 0, axis_len, 0, 0, color='red',   linewidth=2, arrow_length_ratio=0.2)
            ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', linewidth=2, arrow_length_ratio=0.2)
            ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue',  linewidth=2, arrow_length_ratio=0.2)
            ax.text(axis_len + 0.02, 0, 0, 'X', color='red',   fontsize=9, fontweight='bold')
            ax.text(0, axis_len + 0.02, 0, 'Y', color='green', fontsize=9, fontweight='bold')
            ax.text(0, 0, axis_len + 0.02, 'Z', color='blue',  fontsize=9, fontweight='bold')

            ax.legend(loc='upper right', fontsize=8)
            plt.pause(1.0 / UPDATE_HZ)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
