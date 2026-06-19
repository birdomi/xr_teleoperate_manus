import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kinpy as kp
import transform3d as t3d
from scipy.spatial.transform import Rotation as R
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial

def load_episode(episode_path):
    episode_path = os.path.abspath(episode_path)
    data_json_path = os.path.join(episode_path, 'data.json')
    
    if not os.path.exists(data_json_path):
        print(f"Error: {data_json_path} not found.")
        return None

    with open(data_json_path, 'r') as f:
        data_info = json.load(f)
    
    print(f"Loaded episode info: {data_info.get('info', {})}")
    
    frames = []
    for frame in data_info.get('data', []):
        frame_data = {
            'idx': frame['idx'],
            'colors': {},
            'tactiles': {},
            'states': frame.get('states', {})
        }
        
        # Load colors
        for cam_name, rel_path in frame['colors'].items():
            frame_data['colors'][cam_name] = os.path.join(episode_path, rel_path)
            
        # Load tactiles
        if frame.get('tactiles'):
            for sensor_name, sensor_value in frame['tactiles'].items():
                frame_data['tactiles'][sensor_name] = sensor_value
        
        frames.append(frame_data)
        
    return frames

def compute_hand_fk(urdf_path, joint_values, side='left'):
    chain = kp.build_chain_from_urdf(open(urdf_path, 'rb').read())
    
    prefix = f"{side}_"
    
    joint_map = {
        f'{prefix}thumb_metacarpal_joint': joint_values[0],
        f'{prefix}thumb_proximal_joint': joint_values[1],
        f'{prefix}index_proximal_joint': joint_values[2],
        f'{prefix}middle_proximal_joint': joint_values[3],
        f'{prefix}ring_proximal_joint': joint_values[4],
        f'{prefix}pinky_proximal_joint': joint_values[5],
    }
    
    joint_map[f'{prefix}thumb_distal_joint'] = joint_values[1] * 1.0
    joint_map[f'{prefix}index_distal_joint'] = joint_values[2] * 1.155
    joint_map[f'{prefix}middle_distal_joint'] = joint_values[3] * 1.155
    joint_map[f'{prefix}ring_distal_joint'] = joint_values[4] * 1.155
    joint_map[f'{prefix}pinky_distal_joint'] = joint_values[5] * 1.155

    ret = chain.forward_kinematics(joint_map)
    return ret


def _render_single_frame(frame, left_urdf, right_urdf):
    """Render a single frame and return the resulting image as a numpy array."""
    fig = plt.figure(figsize=(15, 8))
    ax_left = fig.add_subplot(121, projection='3d')
    ax_right = fig.add_subplot(122, projection='3d')
    plt.subplots_adjust(left=0.05, right=0.75, wspace=0.1)
    axes_dict = {'left': ax_left, 'right': ax_right}

    # Load the real image from colors dict (first one)
    real_img = None
    if frame.get('colors'):
        cam_name = list(frame['colors'].keys())[0]
        img_path = frame['colors'][cam_name]
        if os.path.exists(img_path):
            real_img = cv2.imread(img_path)
            if real_img is not None:
                h, w = real_img.shape[:2]
                target_h = 800
                target_w = int(w * (target_h / h))
                real_img = cv2.resize(real_img, (target_w, target_h))

    sides = [
        {
            'name': 'left',
            'urdf': left_urdf,
            'ee_key': 'left_ee',
            'offset': np.array([0, 0, 0]),
            'rotation': R.from_euler('z', 0, degrees=True)
        },
        {
            'name': 'right',
            'urdf': right_urdf,
            'ee_key': 'right_ee',
            'offset': np.array([0, 0, 0]),
            'rotation': R.from_euler('z', 180, degrees=True)
        }
    ]

    for side_info in sides:
        side = side_info['name']
        ee_key = side_info['ee_key']
        urdf = side_info['urdf']
        offset = side_info['offset']
        base_rot = side_info['rotation']
        ax = axes_dict[side]

        if ee_key not in frame.get('states', {}):
            continue

        qpos_data = frame['states'][ee_key]['qpos']
        if len(qpos_data) == 0:
            continue

        hand_qpos = np.array(qpos_data)

        try:
            fk_res = compute_hand_fk(urdf, hand_qpos, side=side)
        except Exception as e:
            print(f"FK failed for {side} hand frame {frame['idx']}: {e}")
            continue

        prefix = f"{side}_"
        finger_chains = [
            [f'{prefix}base_link', f'{prefix}thumb_metacarpal_link', f'{prefix}thumb_proximal_link', f'{prefix}thumb_distal_link', f'{prefix}thumb_tip_link'],
            [f'{prefix}base_link', f'{prefix}index_proximal_link', f'{prefix}index_distal_link', f'{prefix}index_tip_link'],
            [f'{prefix}base_link', f'{prefix}middle_proximal_link', f'{prefix}middle_distal_link', f'{prefix}middle_tip_link'],
            [f'{prefix}base_link', f'{prefix}ring_proximal_link', f'{prefix}ring_distal_link', f'{prefix}ring_tip_link'],
            [f'{prefix}base_link', f'{prefix}pinky_proximal_link', f'{prefix}pinky_distal_link', f'{prefix}pinky_tip_link']
        ]

        for chain_links in finger_chains:
            xs, ys, zs = [], [], []
            for link in chain_links:
                if link in fk_res:
                    local_pos = fk_res[link].pos
                    pos = offset + base_rot.apply(local_pos)
                    xs.append(pos[0])
                    ys.append(-pos[1])
                    zs.append(pos[2])
            ax.plot(xs, zs, ys, 'o-', linewidth=1.0, markersize=3, label=f'{side} hand' if chain_links == finger_chains[0] else "")

        origin_pos = offset
        ax.scatter([origin_pos[0]], [origin_pos[2]], [-origin_pos[1]], marker='^', s=5, label=f'{side} Base')

        tactile_data = frame.get('tactiles', {}).get(ee_key)
        if tactile_data is not None and len(tactile_data) == 20:
            finger_tips = [
                f'{prefix}thumb_touch_link',
                f'{prefix}index_touch_link',
                f'{prefix}middle_touch_link',
                f'{prefix}ring_touch_link',
                f'{prefix}pinky_touch_link'
            ]

            scale_factor = 0.001

            for i, tip_name in enumerate(finger_tips):
                if tip_name in fk_res:
                    transform = fk_res[tip_name]
                    local_pos = transform.pos
                    pos = offset + base_rot.apply(local_pos)

                    w, x, y, z = transform.rot
                    r_local = R.from_quat([x, y, z, w])
                    r_world = base_rot * r_local

                    direction = r_world.as_matrix()[:, 0]
                    shear_direction_vector = r_world.as_matrix()[:, 2]

                    normal_force = tactile_data[i*4]
                    shear_force = tactile_data[i*4 + 1]
                    shear_dir_val = tactile_data[i*4 + 2]
                    shear_dir_val = -1 if shear_dir_val == 65535 else shear_dir_val

                    if shear_dir_val != -1:
                        theta_rad = np.deg2rad(shear_dir_val)
                        rot_vec = direction * theta_rad
                        r_shear = R.from_rotvec(rot_vec)
                        real_shear_vec = r_shear.apply(shear_direction_vector)

                        shear_len_scaled = shear_force * scale_factor
                        if shear_len_scaled > 0.001:
                            ax.quiver(
                                pos[0], pos[2], -pos[1],
                                -real_shear_vec[0], -real_shear_vec[2], real_shear_vec[1],
                                length=shear_len_scaled, normalize=True, color='b'
                            )

                    length = normal_force * scale_factor
                    if length > 0.001:
                        ax.quiver(
                            pos[0], pos[2], -pos[1],
                            -direction[0], -direction[2], direction[1],
                            length=length, normalize=True, color='r'
                        )

                    proximity_val = tactile_data[i*4 + 3]
                    prox_scale = 0.001
                    if proximity_val > 0:
                        ax.scatter(
                            [pos[0]], [pos[2]], [-pos[1]],
                            s=proximity_val * prox_scale,
                            c='lime',
                            marker='o',
                            alpha=0.6,
                            label='Proximity' if i == 0 and side == 'left' else ""
                        )

    # Print state values on the side
    text_lines = [f"Frame: {frame['idx']}"]
    for side_info in sides:
        side = side_info['name']
        ee_key = side_info['ee_key']
        text_lines.append(f"\n--- {side.upper()} HAND ---")

        if ee_key in frame.get('states', {}):
            qpos_data = frame['states'][ee_key]['qpos']
            if len(qpos_data) > 0:
                text_lines.append("Joints (qpos):")
                text_lines.append(np.array2string(np.array(qpos_data), precision=6, separator=', ', suppress_small=False))

        tactile_data = frame.get('tactiles', {}).get(ee_key)
        if tactile_data is not None and len(tactile_data) == 20:
            text_lines.append("Tactile (N, S, Dir, Prox):")
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            for i, fn in enumerate(finger_names):
                n_f = tactile_data[i*4]
                s_f = tactile_data[i*4 + 1]
                s_d = tactile_data[i*4 + 2]
                p_v = tactile_data[i*4 + 3]
                text_lines.append(f"  {fn}: {n_f:.1f}, {s_f:.1f}, {s_d:.0f}, {p_v:.0f}")

    fig.text(0.78, 0.5, '\n'.join(text_lines), fontsize=8, va='center', family='monospace')

    for view_side, ax_side in axes_dict.items():
        ax_side.set_xlim([-.1, .2])
        ax_side.set_ylim([-.1, .2])
        ax_side.set_zlim([-.2, .3])
        ax_side.set_axis_off()
        ax_side.set_title(f"{view_side.capitalize()} Hand Pose (Frame {frame['idx']})")
        ax_side.legend()

    # Render figure to image
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    if real_img is not None:
        if img.shape[0] != real_img.shape[0]:
            real_img = cv2.resize(real_img, (int(real_img.shape[1] * (img.shape[0] / real_img.shape[0])), img.shape[0]))
        img = np.hstack((real_img, img))

    print(f"Processed frame {frame['idx']}")
    return frame['idx'], img


def visualize_dual_hand(frames, left_urdf, right_urdf, start_idx=None, end_idx=None, output_dir='dual_hand_viz', video_name='dual_hand_video'):
    if not frames:
        return

    # Filter frames based on start_idx and end_idx
    filtered_frames = []
    for frame in frames:
        idx = frame['idx']
        if start_idx is not None and idx < start_idx:
            continue
        if end_idx is not None and idx > end_idx:
            continue
        filtered_frames.append(frame)

    frames = filtered_frames

    if not frames:
        print("No frames to visualize after filtering.")
        return

    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, f'{video_name}.mp4')
    fps = 10

    # Use multiprocessing to render frames in parallel
    num_workers = min(cpu_count(), len(frames))
    print(f"Rendering {len(frames)} frames using {num_workers} workers...")

    render_fn = partial(_render_single_frame, left_urdf=left_urdf, right_urdf=right_urdf)

    with Pool(processes=num_workers) as pool:
        results = pool.map(render_fn, frames)

    # Sort by frame index to ensure correct order
    results.sort(key=lambda x: x[0])

    # Write video sequentially
    video_writer = None
    for idx, img in results:
        if img is None:
            continue
        if video_writer is None:
            height, width, _ = img.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        video_writer.write(img)

    if video_writer is not None:
        video_writer.release()
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize dual hand episodes from a parent directory")
    parser.add_argument('--episode_dir', type=str, default='0305/basket', help='Path to parent directory containing episodes, or a specific episode directory')
    parser.add_argument('--left_urdf', type=str, default='/home/goodman/Downloads/urdf/revo2_left_hand.urdf', help='Path to left hand URDF')
    parser.add_argument('--right_urdf', type=str, default='/home/goodman/Downloads/urdf/revo2_right_hand.urdf', help='Path to right hand URDF')
    parser.add_argument('--start_idx', type=int, default=None, help='Starting frame index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None, help='Ending frame index (inclusive)')
    parser.add_argument('--output_dir', type=str, default='dual_hand_viz', help='Output directory for videos and frames')
    args = parser.parse_args()
    
    parent_dir = args.episode_dir
    if not os.path.exists(parent_dir):
        print(f"Error: {parent_dir} not found.")
        exit(1)

    # Check if parent_dir itself is an episode directory
    if os.path.exists(os.path.join(parent_dir, 'data.json')):
        episode_dirs = [parent_dir]
    else:
        # Find all subdirectories that contain 'data.json'
        episode_dirs = []
        for d in sorted(os.listdir(parent_dir)):
            full_path = os.path.join(parent_dir, d)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'data.json')):
                episode_dirs.append(full_path)
                
    if not episode_dirs:
        print(f"No valid episode directories found in {parent_dir}.")
        exit(1)

    for ep_dir in episode_dirs:
        print(f"\n--- Processing episode: {ep_dir} ---")
        frames = load_episode(ep_dir)
        
        if frames:
            print(f"Successfully loaded {len(frames)} frames for {ep_dir}.")
            norm_ep_dir = os.path.normpath(ep_dir)
            ep_name = os.path.basename(norm_ep_dir)
            parent_name = os.path.basename(os.path.dirname(norm_ep_dir))
            
            video_filename = f"{parent_name}_{ep_name}" if parent_name else ep_name
            
            # Use the episode folder name as the video name
            visualize_dual_hand(
                frames, 
                args.left_urdf, 
                args.right_urdf, 
                start_idx=args.start_idx, 
                end_idx=args.end_idx, 
                output_dir=args.output_dir, 
                video_name=video_filename
            )