import os
import json
import cv2
import time
import rerun as rr
import rerun.blueprint as rrb
from datetime import datetime
os.environ["RUST_LOG"] = "error"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

class RerunEpisodeReader:
    def __init__(self, task_dir = ".", json_file="data.json"):
        self.task_dir = task_dir
        self.json_file = json_file

    def return_episode_data(self, episode_idx):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_idx} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        episode_data = []

        # Loop over the data entries and process each one
        for item_data in json_file['data']:
            # Process images and other data
            colors = self._process_images(item_data, 'colors', episode_dir)
            depths = self._process_images(item_data, 'depths', episode_dir)
            audios = self._process_audio(item_data, 'audios', episode_dir)

            # Append the data in the item_data list
            episode_data.append(
                {
                    'idx': item_data.get('idx', 0),
                    'colors': colors,
                    'depths': depths,
                    'states': item_data.get('states', {}),
                    'actions': item_data.get('actions', {}),
                    'tactiles': item_data.get('tactiles', {}),
                    'audios': audios,
                }
            )

        return episode_data

    def _process_images(self, item_data, data_type, dir_path):
        images = {}

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images[key] = image
        return images

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_data = {}
        dir_path = os.path.join(episode_dir, data_type)

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    pass  # Handle audio data if needed
class RerunLogger:
    def __init__(self, prefix = "", IdxRangeBoundary = 30, memory_limit = None, show_tactile=False):
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        self.show_tactile = show_tactile
        rr.init(datetime.now().strftime("Runtime_%Y%m%d_%H%M%S"))
        if memory_limit:
            rr.spawn(memory_limit = memory_limit, hide_welcome_screen = True)
        else:
            rr.spawn(hide_welcome_screen = True)

        # Set up blueprint for live visualization
        if self.IdxRangeBoundary:
            self.setup_blueprint()

    def setup_blueprint(self):
        views = []

        data_plot_paths = [
                           f"{self.prefix}left_arm", 
                           f"{self.prefix}right_arm", 
                           f"{self.prefix}left_ee", 
                           f"{self.prefix}right_ee"
        ]
        for plot_path in data_plot_paths:
            view = rrb.TimeSeriesView(
                origin = plot_path,
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "idx",
                        start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                        end = rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
                plot_legend = rrb.PlotLegend(visible = True),
            )
            views.append(view)

        # image_plot_paths = [
        #                     f"{self.prefix}colors/color_0",
        #                     f"{self.prefix}colors/color_1",
        #                     f"{self.prefix}colors/color_2",
        #                     f"{self.prefix}colors/color_3"
        # ]
        # for plot_path in image_plot_paths:
        #     view = rrb.Spatial2DView(
        #         origin = plot_path,
        #         time_ranges=[
        #             rrb.VisibleTimeRange(
        #                 "idx",
        #                 start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
        #                 end = rrb.TimeRangeBoundary.cursor_relative(),
        #             )
        #         ],
        #     )
        #     views.append(view)

        if getattr(self, 'show_tactile', True):
            tactile_plot_paths = [
                f"{self.prefix}tactiles/left_ee",
                f"{self.prefix}tactiles/right_ee"
            ]
            for plot_path in tactile_plot_paths:
                view = rrb.Spatial2DView(
                    origin = plot_path,
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "idx",
                            start = rrb.TimeRangeBoundary.cursor_relative(seq = -1),
                            end = rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                )
                views.append(view)

        grid = rrb.Grid(contents = views,
                        grid_columns=2,               
                        column_shares=[1, 1],
                        row_shares=[1, 1], 
        )
        views.append(rr.blueprint.SelectionPanel(state=rrb.PanelState.Collapsed))
        views.append(rr.blueprint.TimePanel(state=rrb.PanelState.Collapsed))
        rr.send_blueprint(grid)

    def _render_tactile_image(self, tactile_vals, hand):
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

        # logger_mp.info(tactile_vals)

        proximities = []
        normal_forces = []
        tangential_us = []
        tangential_vs = []
        
        for i in range(5):
            # Ensure it is a scalar, as tactile_vals might be a nested array or numpy type
            normal_force = float(tactile_vals[i * 4])
            tangential_force = float(tactile_vals[i * 4 + 1])
            tangential_dir = float(tactile_vals[i * 4 + 2])
            val = float(tactile_vals[i * 4 + 3])
            
            normal_forces.append(normal_force)
            proximities.append(val)
            
            # Compute U, V for quiver plot
            # Convert direction (0-365) to radians. 
            # Make 0 degrees point up (12 o'clock, which is +Y in pyplot), 
            # and positive angles go clockwise (or counter-clockwise depending on convention, assuming clockwise here).
            # Start at 90 degrees (pi/2) and subtract the angle.
            if tangential_dir != 65535:
                theta = np.deg2rad(90 - tangential_dir)
            else:
                theta = 0
                
            # Scale tangential force so vectors are visibly scaled (adjust 1000.0 as needed)
            r = tangential_force / 1000.0 if tangential_dir != 65535 else 0
            tangential_us.append(r * np.cos(theta))
            tangential_vs.append(r * np.sin(theta))
        # print(theta, tangential_dir)
            
        # Create a white image (400x400, 3 channels)
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Approximate finger tip coordinates (x, y) relative to palm center for RIGHT hand
        xs = [-1.5, -0.6,  0.0,  0.6,  1.3]
        
        # Invert xs for left hand so it looks mirrored (thumb on the right)
        if 'left' in hand.lower():
            xs = [-x for x in xs]
            
        ys = [ 0.5,  2.0,  2.2,  1.9,  1.2]
        
        # Normalize sizes for scatter plot (max area ~ 2000)
        # Add a minimum size so fingers are always visible
        sizes = [max(100.0, float((p / 450000.0) * 100.0)) for p in proximities]
        normal_sizes = [max(0.0, float((n / 25000.0) * 5000.0)) for n in normal_forces]
        
        # Add title
        cv2.putText(img, 'Tactile Info (Blue:Prox, Red:Norm, Grn:Tang)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        
        for i in range(5):
            # Transform coordinates to image space
            # x: -2.5 to 2.5 -> 0 to 400
            # y: -0.5 to 3.5 -> 400 to 0
            cx = int((xs[i] + 2.5) / 5.0 * 400)
            cy = int((3.5 - ys[i]) / 4.0 * 400)
            
            # Radii for circles from area 's'
            r_prox = max(1, int(np.sqrt(sizes[i]) * 0.8))
            r_norm = max(1, int(np.sqrt(normal_sizes[i]) * 0.8))
            
            # Draw proximity circles (skyblue: BGR 235, 206, 135)
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), r_prox, (235, 206, 135), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            # Draw normal force circles (red: BGR 0, 0, 255)
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), r_norm, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
            
            # Draw small center points for reference ('x' mark)
            cv2.line(img, (cx-3, cy-3), (cx+3, cy+3), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.line(img, (cx-3, cy+3), (cx+3, cy-3), (0, 0, 0), 1, cv2.LINE_AA)
            
            # Draw tangential force vectors (green arrows)
            if tangential_us[i] != 0 or tangential_vs[i] != 0:
                end_x = int((xs[i] + tangential_us[i] + 2.5) / 5.0 * 400)
                end_y = int((3.5 - (ys[i] + tangential_vs[i])) / 4.0 * 400)
                cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 128, 0), 2, tipLength=0.2, line_type=cv2.LINE_AA)
            
            # Add labels
            name_size = cv2.getTextSize(finger_names[i], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(img, finger_names[i], (cx - name_size[0]//2, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
            prox_str = f"{proximities[i]:.0f}"
            prox_size = cv2.getTextSize(prox_str, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0]
            cv2.putText(img, prox_str, (cx - prox_size[0]//2, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 0, 0), 1, cv2.LINE_AA)
            
            norm_str = f"{normal_forces[i]:.0f}"
            norm_size = cv2.getTextSize(norm_str, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0]
            cv2.putText(img, norm_str, (cx - norm_size[0]//2, cy - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 255), 1, cv2.LINE_AA)
            
        # Convert BGR to RGB (OpenCV uses BGR, we need RGB for rerun)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def log_item_data(self, item_data: dict):
        rr.set_time_sequence("idx", item_data.get('idx', 0))

        # Log states
        states = item_data.get('states', {}) or {}
        for part, state_info in states.items():
            if part != "body" and state_info:
                values = state_info.get('qpos', [])
                for idx, val in enumerate(values):
                    rr.log(f"{self.prefix}{part}/states/qpos/{idx}", rr.Scalar(val))

        # Log actions
        actions = item_data.get('actions', {}) or {}
        for part, action_info in actions.items():
            if part != "body" and action_info:
                values = action_info.get('qpos', [])
                for idx, val in enumerate(values):
                    rr.log(f"{self.prefix}{part}/actions/qpos/{idx}", rr.Scalar(val))

        # # Log colors (images)
        # colors = item_data.get('colors', {}) or {}
        # for color_key, color_val in colors.items():
        #     if color_val is not None:
        #         rr.log(f"{self.prefix}colors/{color_key}", rr.Image(color_val))

        # # Log depths (images)
        # depths = item_data.get('depths', {}) or {}
        # for depth_key, depth_val in depths.items():
        #     if depth_val is not None:
        #         # rr.log(f"{self.prefix}depths/{depth_key}", rr.Image(depth_val))
        #         pass # Handle depth if needed

        # # Log tactile if needed
        # tactiles = item_data.get('tactiles', {}) or {}
        # for hand, tactile_vals in tactiles.items():
        #     if tactile_vals is not None:
        #         pass # Handle tactile if needed

        # Log tactile if needed
        if self.show_tactile:
            tactiles = item_data.get('tactiles', {}) or {}
            
            for hand, tactile_vals in tactiles.items():
                tactile_img = self._render_tactile_image(tactile_vals, hand)
                rr.log(f"{self.prefix}tactiles/{hand}", rr.Image(tactile_img))

        # # Log audios if needed
        # audios = item_data.get('audios', {}) or {}
        # for audio_key, audio_val in audios.items():
        #     if audio_val is not None:
        #         pass  # Handle audios if needed

    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)


if __name__ == "__main__":
    import gdown
    import zipfile
    import os
    import logging_mp
    logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)

    zip_file = "rerun_testdata.zip"
    zip_file_download_url = "https://drive.google.com/file/d/1f5UuFl1z_gaByg_7jDRj1_NxfJZh2evD/view?usp=sharing"
    unzip_file_output_dir = "./testdata"
    if not os.path.exists(os.path.join(unzip_file_output_dir, "episode_0006")):
        if not os.path.exists(zip_file):
            file_id = zip_file_download_url.split('/')[5]
            gdown.download(id=file_id, output=zip_file, quiet=False)
            logger_mp.info("download ok.")
        if not os.path.exists(unzip_file_output_dir):
            os.makedirs(unzip_file_output_dir)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_file_output_dir)
        logger_mp.info("uncompress ok.")
        os.remove(zip_file)
        logger_mp.info("clean file ok.")
    else:
        logger_mp.info("rerun_testdata exits.")


    episode_reader = RerunEpisodeReader(task_dir = unzip_file_output_dir)
    # TEST EXAMPLE 1 : OFFLINE DATA TEST
    user_input = input("Please enter the start signal (enter 'off' or 'on' to start the subsequent program):\n")
    if user_input.lower() == 'off':
        episode_data6 = episode_reader.return_episode_data(6)
        logger_mp.info("Starting offline visualization...")
        offline_logger = RerunLogger(prefix="offline/")
        offline_logger.log_episode_data(episode_data6)
        logger_mp.info("Offline visualization completed.")

    # TEST EXAMPLE 2 : ONLINE DATA TEST, SLIDE WINDOW SIZE IS 60, MEMORY LIMIT IS 50MB
    if user_input.lower() == 'on':
        episode_data8 = episode_reader.return_episode_data(8)
        logger_mp.info("Starting online visualization with fixed idx size...")
        online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit='50MB')
        for item_data in episode_data8:
            online_logger.log_item_data(item_data)
            time.sleep(0.033) # 30hz
        logger_mp.info("Online visualization completed.")


    # # TEST DATA OF data_dir
    # data_dir = "./data"
    # episode_data_number = 10
    # episode_reader2 = RerunEpisodeReader(task_dir = data_dir)
    # user_input = input("Please enter the start signal (enter 'on' to start the subsequent program):\n")
    # episode_data8 = episode_reader2.return_episode_data(episode_data_number)
    # if user_input.lower() == 'on':
    #     # Example 2: Offline Visualization with Fixed Time Window
    #     logger_mp.info("Starting offline visualization with fixed idx size...")
    #     online_logger = RerunLogger(prefix="offline/", IdxRangeBoundary = 60)
    #     for item_data in episode_data8:
    #         online_logger.log_item_data(item_data)
    #         time.sleep(0.033) # 30hz
    #     logger_mp.info("Offline visualization completed.")