import sys
import os
import glob
import numpy as np
import re
import time
import json
from collections import deque
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QGridLayout, QLabel, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFileDialog, QSlider, QSpinBox,
                            QCheckBox, QProgressBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
import colorcet
import cv2

# 기존 코드에서 가져온 데이터 정의
data_sheet = [
    ("Pinky fingertip tactile data", 3000, 18, (3, 3), "fingerone_tip_touch"),
    ("Pinky top tactile data", 3018, 192, (12, 8), "fingerone_top_touch"),
    ("Pinky palm tactile data", 3210, 160, (10, 8), "fingerone_palm_touch"),
    ("Ring fingertip tactile data", 3370, 18, (3, 3), "fingertwo_tip_touch"),
    ("Ring top tactile data", 3388, 192, (12, 8), "fingertwo_top_touch"),
    ("Ring palm tactile data", 3580, 160, (10, 8), "fingertwo_palm_touch"),
    ("Middle fingertip tactile data", 3740, 18, (3, 3), "fingerthree_tip_touch"),
    ("Middle top tactile data", 3758, 192, (12, 8), "fingerthree_top_touch"),
    ("Middle palm tactile data", 3950, 160, (10, 8), "fingerthree_palm_touch"),
    ("Index fingertip tactile data", 4110, 18, (3, 3), "fingerfour_tip_touch"),
    ("Index top tactile data", 4128, 192, (12, 8), "fingerfour_top_touch"),
    ("Index palm tactile data", 4320, 160, (10, 8), "fingerfour_palm_touch"),
    ("Thumb fingertip tactile data", 4480, 18, (3, 3), "fingerfive_tip_touch"),
    ("Thumb top tactile data", 4498, 192, (12, 8), "fingerfive_top_touch"),
    ("Thumb middle tactile data", 4690, 18, (3, 3), "fingerfive_middle_touch"),
    ("Thumb palm tactile data", 4708, 192, (12, 8), "fingerfive_palm_touch"),
    ("Palm tactile data", 4900, 224, (14, 8), "palm_touch")
]

tactile_field_sizes = [9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 9, 96, 112]

class NPYDataHandler:
    """NPY 파일과 이미지 파일, JSON 파일을 읽어서 데이터를 처리하는 클래스"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tactile_dir = os.path.join(data_dir, 'tactiles')
        self.colors_dir = os.path.join(data_dir, 'colors')
        self.json_file = os.path.join(data_dir, 'data.json')
        
        self.file_pairs = []
        self.image_sets = []
        self.json_data = []
        self.current_index = 0
        
        self.load_file_pairs()
        self.load_image_sets()
        self.load_json_data()
    
    def load_json_data(self):
        """JSON 파일 로드"""
        if not os.path.exists(self.json_file):
            print(f"JSON file {self.json_file} does not exist")
            return
        
        try:
            with open(self.json_file, 'r') as f:
                json_content = json.load(f)
            
            # JSON 데이터에서 data 배열 추출
            if 'data' in json_content:
                self.json_data = json_content['data']
                print(f"Loaded {len(self.json_data)} JSON data entries")
            else:
                print("No 'data' field found in JSON file")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
    
    def load_file_pairs(self):
        """디렉토리에서 tactile .npy 파일 쌍들을 로드"""
        if not os.path.exists(self.tactile_dir):
            print(f"Tactile directory {self.tactile_dir} does not exist")
            return
        
        left_pattern = os.path.join(self.tactile_dir, "tactile_*_left_ee.npy")
        right_pattern = os.path.join(self.tactile_dir, "tactile_*_right_ee.npy")
        
        left_files = glob.glob(left_pattern)
        right_files = glob.glob(right_pattern)
        
        def extract_number(filename):
            match = re.search(r'tactile_(\d+)_', filename)
            return int(match.group(1)) if match else 0
        
        left_files.sort(key=extract_number)
        right_files.sort(key=extract_number)
        
        # 좌우 파일 쌍 생성
        left_dict = {extract_number(f): f for f in left_files}
        right_dict = {extract_number(f): f for f in right_files}
        
        common_numbers = set(left_dict.keys()) & set(right_dict.keys())
        
        for num in sorted(common_numbers):
            self.file_pairs.append({
                'left': left_dict[num],
                'right': right_dict[num],
                'index': num
            })
        
        print(f"Loaded {len(self.file_pairs)} tactile file pairs")
    
    def load_image_sets(self):
        """디렉토리에서 이미지 파일 세트들을 로드"""
        if not os.path.exists(self.colors_dir):
            print(f"Colors directory {self.colors_dir} does not exist")
            return
        
        # 패턴: 000000_color_0.jpg, 000000_color_1.jpg, 000000_color_2.jpg, 000000_color_3.jpg
        color_pattern = os.path.join(self.colors_dir, "*_color_*.jpg")
        color_files = glob.glob(color_pattern)
        
        def extract_frame_number(filename):
            match = re.search(r'(\d{6})_color_', filename)
            return int(match.group(1)) if match else 0
        
        # 프레임 번호별로 그룹화
        frame_groups = {}
        for file in color_files:
            frame_num = extract_frame_number(file)
            if frame_num not in frame_groups:
                frame_groups[frame_num] = {}
            
            # color_X.jpg에서 X 추출
            match = re.search(r'color_(\d)\.jpg', file)
            if match:
                cam_idx = int(match.group(1))
                frame_groups[frame_num][cam_idx] = file
        
        # 4개 카메라가 모두 있는 프레임만 선택
        for frame_num in sorted(frame_groups.keys()):
            cameras = frame_groups[frame_num]
            if all(i in cameras for i in [0, 1, 2, 3]):  # 0, 1, 2, 3 모두 있는지 확인
                self.image_sets.append({
                    'index': frame_num,
                    'stereo_left': cameras[0],    # color_0
                    'stereo_right': cameras[1],   # color_1
                    'wrist_left': cameras[2],     # color_2
                    'wrist_right': cameras[3]     # color_3
                })
        
        print(f"Loaded {len(self.image_sets)} image sets")
    
    def parse_tactile_data(self, data, hand='left'):
        """Tactile 데이터를 각 필드별로 분리"""
        fields = {}
        data_idx = 0
        
        for i, (name, addr, length, size, var) in enumerate(data_sheet):
            field_size = tactile_field_sizes[i]
            
            if data_idx + field_size <= len(data):
                field_data = data[data_idx:data_idx + field_size]
                # 2D matrix로 reshape
                matrix = np.array(field_data, dtype=np.float64).reshape(size)
                fields[var] = matrix
                data_idx += field_size
            else:
                # 데이터가 부족한 경우 0으로 채움
                fields[var] = np.zeros(size)
        
        return fields
    
    def read_current_frame(self):
        """현재 프레임의 tactile 데이터 읽기"""
        if not self.file_pairs or self.current_index >= len(self.file_pairs):
            return None
        
        try:
            current_pair = self.file_pairs[self.current_index]
            left_data = np.load(current_pair['left'])
            right_data = np.load(current_pair['right'])
            
            left_fields = self.parse_tactile_data(left_data, 'left')
            right_fields = self.parse_tactile_data(right_data, 'right')
            
            return {
                'left': left_fields,
                'right': right_fields,
                'index': current_pair['index'],
                'time': self.current_index / 30.0  # 30Hz 가정
            }
        except Exception as e:
            print(f"Error reading tactile frame {self.current_index}: {e}")
            return None
    
    def read_current_images(self):
        """현재 프레임의 이미지 데이터 읽기 - skimage 사용"""
        if not self.image_sets or self.current_index >= len(self.image_sets):
            return None
        
        try:
            current_images = self.image_sets[self.current_index]
            
            # 이미지 파일들 로드
            images = {}
            for key, filepath in current_images.items():
                if key != 'index' and os.path.exists(filepath):
                    try:
                        # skimage로 이미지 로드 (가장 안전한 방법)
                        from skimage import io
                        img_array = io.imread(filepath)
                        
                        # RGB 형식 확인 및 변환
                        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                            images[key] = img_array
                        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                            # RGBA를 RGB로 변환
                            images[key] = img_array[:, :, :3]
                        else:
                            print(f"Unexpected image format for {filepath}: {img_array.shape}")
                            
                    except ImportError:
                        # skimage가 없으면 matplotlib 사용
                        try:
                            import matplotlib.image as mpimg
                            img_array = mpimg.imread(filepath)
                            # 0-1 범위를 0-255로 변환 (필요한 경우)
                            if img_array.max() <= 1.0:
                                img_array = (img_array * 255).astype(np.uint8)
                            images[key] = img_array
                        except Exception as fallback_error:
                            print(f"Error loading image {filepath} with fallback: {fallback_error}")
                            continue
                    except Exception as img_error:
                        print(f"Error loading image {filepath}: {img_error}")
                        continue
            
            return {
                'images': images,
                'index': current_images['index'],
                'time': self.current_index / 30.0
            }
        except Exception as e:
            print(f"Error reading images frame {self.current_index}: {e}")
            return None
    
    def read_current_json_data(self):
        """현재 프레임의 JSON 데이터 읽기"""
        if not self.json_data or self.current_index >= len(self.json_data):
            return None
        
        try:
            current_data = self.json_data[self.current_index]
            return {
                'data': current_data,
                'index': current_data.get('idx', self.current_index),
                'time': self.current_index / 30.0  # 30Hz 가정
            }
        except Exception as e:
            print(f"Error reading JSON frame {self.current_index}: {e}")
            return None
    
    def set_frame(self, frame_index):
        """특정 프레임으로 이동"""
        max_index = min(len(self.file_pairs), len(self.image_sets), len(self.json_data)) - 1
        if 0 <= frame_index <= max_index:
            self.current_index = frame_index
            return True
        return False
    
    def next_frame(self):
        """다음 프레임으로 이동"""
        max_index = min(len(self.file_pairs), len(self.image_sets), len(self.json_data)) - 1
        if self.current_index < max_index:
            self.current_index += 1
            return True
        return False
    
    def prev_frame(self):
        """이전 프레임으로 이동"""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False
    
    def get_total_frames(self):
        """총 프레임 수 반환"""
        return min(len(self.file_pairs), len(self.image_sets), len(self.json_data))

class CameraImageTab(QWidget):
    """카메라 이미지를 표시하는 탭 (2x2 레이아웃)"""
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 제목 라벨
        self.title_label = QLabel("CAMERA IMAGES")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: green;")
        self.layout.addWidget(self.title_label)
        
        # 2x2 그리드 레이아웃
        self.grid_layout = QGridLayout()
        self.layout.addLayout(self.grid_layout)
        
        self.create_image_widgets()
    
    # CameraImageTab 클래스의 create_image_widgets 메소드를 다음과 같이 수정하세요:

    def create_image_widgets(self):
        """2x2 이미지 위젯 생성"""
        self.image_labels = {}
        
        # 이미지 위젯 정의 (위치와 제목)
        image_layout = [
            ('stereo_left', 0, 0, 'Head Camera Left'),
            ('stereo_right', 0, 1, 'Head Camera Right'), 
            ('wrist_left', 1, 0, 'Wrist Camera Left'),
            ('wrist_right', 1, 1, 'Wrist Camera Right')
        ]
        
        for key, row, col, title in image_layout:
            # 컨테이너 위젯
            container = QWidget()
            container.setStyleSheet("border: 2px solid #ccc; background-color: #f9f9f9;")
            container_layout = QVBoxLayout()
            container.setLayout(container_layout)
            
            # 제목 라벨
            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: bold; font-size: 15px; text-align: center;")
            title_label.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(title_label)
            
            # 이미지 라벨
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("background-color: black; color: white;")
            image_label.setText("No Image")
            
            # 크기 정책을 확장 가능하도록 설정
            from PyQt5.QtWidgets import QSizePolicy
            image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # 최소 크기만 설정 (고정 크기 제거)
            image_label.setMinimumSize(320, 240)  # 최소 크기 설정
            image_label.setScaledContents(False)  # 가로세로 비율 유지를 위해 False로 설정
            
            container_layout.addWidget(image_label)
            self.image_labels[key] = image_label
            
            # 컨테이너도 확장 가능하도록 설정
            container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # 그리드에 추가
            self.grid_layout.addWidget(container, row, col)
    
    def update_images(self, image_data):
        """이미지 업데이트"""
        if image_data is None or 'images' not in image_data:
            return
        
        images = image_data['images']
        
        for key, image_label in self.image_labels.items():
            if key in images:
                img_array = images[key]
                
                # NumPy 배열을 QPixmap으로 변환
                h, w, ch = img_array.shape
                bytes_per_line = ch * w
                
                # RGB에서 Qt 형식으로 변환
                q_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                # 이미지 라벨에 설정 (aspect ratio 유지하며 스케일링)
                scaled_pixmap = pixmap.scaled(
                    image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
            else:
                # 이미지가 없으면 "No Image" 표시
                image_label.clear()

class TactileImageTab(QWidget):
    """Tactile 이미지를 표시하는 탭 (한 손만 표시)"""
    
    def __init__(self, datas=data_sheet, hand='left'):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.hand = hand  # 'left' or 'right'
        
        # 탭 제목 라벨 추가
        self.title_label = QLabel(f"{self.hand.upper()} HAND TACTILE DATA")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;" if hand == 'left' else "font-weight: bold; font-size: 14px; color: red;")
        self.layout.addWidget(self.title_label)
        
        self.grid_layout = QGridLayout()
        self.layout.addLayout(self.grid_layout)
        self.data_sheet = datas
        
        self.create_images()
    
    def create_images(self):
        """이미지 플롯 생성 - 손바닥 시점에서 자연스러운 배치"""
        self.plots = []
        self.color_maps = []
        self.color_bars = []
        
        # 손가락별 데이터 매핑 (data_sheet의 var 이름 기준)
        finger_mapping = {
            'thumb': ['fingerfive_tip_touch', 'fingerfive_top_touch', 'fingerfive_middle_touch', 'fingerfive_palm_touch'],
            'index': ['fingerfour_tip_touch', 'fingerfour_top_touch', None, 'fingerfour_palm_touch'], 
            'middle': ['fingerthree_tip_touch', 'fingerthree_top_touch', None, 'fingerthree_palm_touch'],
            'ring': ['fingertwo_tip_touch', 'fingertwo_top_touch', None, 'fingertwo_palm_touch'],
            'pinky': ['fingerone_tip_touch', 'fingerone_top_touch', None, 'fingerone_palm_touch']
        }
        
        # 손바닥을 보는 시점에서의 배치 정의
        if self.hand == 'left':
            # 왼손: 엄지 -> 검지 -> 중지 -> 약지 -> 새끼 (엄지가 왼쪽)
            finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        else:
            # 오른손: 새끼 -> 약지 -> 중지 -> 검지 -> 엄지 (엄지가 오른쪽)
            finger_order = ['pinky', 'ring', 'middle', 'index', 'thumb']
        
        # 행별 라벨
        row_labels = ['Fingertip', 'Top', 'Middle/Palm', 'Palm']
        
        # 그리드 배치 (4행 5열)
        grid_layout = [
            # 첫 번째 행: fingertip
            [finger_mapping[finger][0] for finger in finger_order],
            # 두 번째 행: top
            [finger_mapping[finger][1] for finger in finger_order],
            # 세 번째 행: middle (thumb만) / palm (나머지 4개 손가락)
            [finger_mapping[finger][2] if finger == 'thumb' else finger_mapping[finger][3] for finger in finger_order],
            # 네 번째 행: palm (thumb만), 나머지는 palm_touch 위치 조정
            [finger_mapping[finger][3] if finger == 'thumb' else 
             ('palm_touch' if (self.hand == 'left' and finger == finger_order[1]) or 
                              (self.hand == 'right' and finger == finger_order[3]) else None) 
             for finger in finger_order]
        ]
        
        # 플롯 생성 및 배치
        for row_idx, row_items in enumerate(grid_layout):
            for col_idx, var_name in enumerate(row_items):
                if var_name is not None:
                    # data_sheet에서 해당 필드 찾기
                    field_info = None
                    for i, (name, addr, length, size, var) in enumerate(self.data_sheet):
                        if var == var_name:
                            field_info = (name, addr, length, size, var, i)
                            break
                    
                    if field_info:
                        name, addr, length, size, var, data_idx = field_info
                        
                        # 그래픽 레이아웃 위젯 생성
                        layout_widget = pg.GraphicsLayoutWidget(show=True)
                        plot_item = layout_widget.addPlot(row=0, col=0)
                        
                        # 제목 설정 (손가락 이름 + 부위)
                        finger_name = finger_order[col_idx].capitalize()
                        part_name = row_labels[row_idx]
                        
                        if var_name == 'palm_touch':
                            plot_title = "Palm Tactile"
                        else:
                            plot_title = f"{finger_name} {part_name}"
                        
                        plot_item.setTitle(plot_title)
                        
                        # 이미지 아이템 생성
                        img_item = pg.ImageItem(np.random.rand(*size) * 100)
                        plot_item.addItem(img_item)
                        
                        # 컬러맵 생성
                        color_map = pg.ColorMap(pos=np.linspace(0, 1, 256), color=colorcet.fire[:256])
                        
                        # 컬러바 생성 (가로 방향)
                        color_bar = pg.ColorBarItem(
                            colorMap=color_map, 
                            values=(0, 100), 
                            width=5, 
                            orientation='h'
                        )
                        layout_widget.addItem(color_bar, row=1, col=0)
                        
                        # 초기 컬러맵 적용
                        img_item.setColorMap(color_map)
                        
                        # 배열에 저장 (data_sheet 순서 인덱스로)
                        while len(self.plots) <= data_idx:
                            self.plots.append(None)
                            self.color_maps.append(None)
                            self.color_bars.append(None)
                        
                        self.plots[data_idx] = img_item
                        self.color_maps[data_idx] = color_map
                        self.color_bars[data_idx] = color_bar
                        
                        # 그리드에 위젯 추가
                        self.grid_layout.addWidget(layout_widget, row_idx, col_idx)
                else:
                    # 빈 공간에는 빈 위젯 추가
                    empty_widget = QWidget()
                    empty_widget.setStyleSheet("background-color: #f0f0f0; border: 1px dashed #ccc;")
                    empty_label = QLabel("N/A")
                    empty_label.setAlignment(Qt.AlignCenter)
                    empty_label.setStyleSheet("color: #888; font-style: italic;")
                    
                    empty_layout = QVBoxLayout()
                    empty_layout.addWidget(empty_label)
                    empty_widget.setLayout(empty_layout)
                    
                    self.grid_layout.addWidget(empty_widget, row_idx, col_idx)
    
    def update_plot(self, data_dict):
        """플롯 업데이트 - 해당 손의 데이터만 표시 (조용히)"""
        if data_dict is None:
            return
        
        # 해당 손의 데이터만 가져오기
        hand_data = data_dict.get(self.hand, {})
        
        if not hand_data:
            return
        
        # 데이터 업데이트 (조용히)
        for i, (name, addr, length, size, var) in enumerate(self.data_sheet):
            if var in hand_data and i < len(self.plots) and self.plots[i] is not None:
                matrix = hand_data[var].copy()
                
                # 이미지 업데이트
                self.plots[i].setImage(matrix, autoLevels=True)
                
                # 레벨 설정
                min_val, max_val = matrix.min(), matrix.max()
                if max_val > min_val:
                    self.plots[i].setLevels((min_val, max_val))
                    if self.color_bars[i] is not None:
                        self.color_bars[i].setLevels((min_val, max_val))
                else:
                    self.plots[i].setLevels((min_val, min_val + 1))
                    if self.color_bars[i] is not None:
                        self.color_bars[i].setLevels((min_val, min_val + 1))
                
                # 컬러맵 적용
                if self.color_maps[i] is not None:
                    self.plots[i].setColorMap(self.color_maps[i])

class TactileCurveTab(QWidget):
    """Tactile 데이터를 시간에 따른 곡선으로 표시하는 탭 (오실로스코프 방식)"""
    
    def __init__(self, datas=data_sheet):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.data_sheet = datas
        
        # 고정된 시간 윈도우 (5초)
        self.time_window = 5.0
        # 샘플링 주파수 (30Hz)
        self.sample_rate = 30.0
        # 총 샘플 수
        self.total_samples = int(self.time_window * self.sample_rate)  # 150 samples
        
        # 고정된 시간 배열 (0 ~ 5초)
        self.time_array = np.linspace(0, self.time_window, self.total_samples)
        
        # 현재 쓰기 위치 (circular buffer index)
        self.write_index = 0
        
        # 데이터 버퍼 초기화 (각 필드별로 좌우 분리)
        self.left_data_buffers = {}
        self.right_data_buffers = {}
        
        for name, addr, length, size, var in self.data_sheet:
            self.left_data_buffers[var] = np.zeros(self.total_samples)
            self.right_data_buffers[var] = np.zeros(self.total_samples)
        
        # 유효한 데이터 개수 추적
        self.valid_data_count = 0
        
        self.create_curves()
    
    def create_curves(self):
        """곡선 그래프 생성"""
        self.grid_layout = QGridLayout()
        self.layout.addLayout(self.grid_layout)
        
        self.plot_widgets = {}
        self.left_curves = {}
        self.right_curves = {}
        
        # 각 필드별로 그래프 생성
        positions = [(i // 4, i % 4) for i in range(len(self.data_sheet))]
        
        for i, (name, addr, length, size, var) in enumerate(self.data_sheet):
            if i < len(positions):
                row, col = positions[i]
                
                plot_widget = pg.PlotWidget()
                plot_widget.setTitle(name)
                plot_widget.setLabel('left', 'Average Value')
                plot_widget.setLabel('bottom', 'Time (s)')
                plot_widget.addLegend()
                plot_widget.showGrid(x=True, y=True, alpha=0.3)
                
                # x축 범위 완전 고정 (절대 변하지 않음)
                plot_widget.setXRange(0, self.time_window, padding=0)
                plot_widget.enableAutoRange(axis='x', enable=False)  # x축 자동 스케일링 비활성화
                plot_widget.setMouseEnabled(x=False, y=True)  # x축 마우스 줌/팬 비활성화
                
                # ViewBox의 x축 범위도 강제로 고정
                view_box = plot_widget.getViewBox()
                view_box.setLimits(xMin=0, xMax=self.time_window)
                
                # 좌우 곡선 생성 (굵은 선으로)
                left_pen = pg.mkPen(color='b', width=2)
                right_pen = pg.mkPen(color='r', width=2)
                left_curve = plot_widget.plot(pen=left_pen, name='Left')
                right_curve = plot_widget.plot(pen=right_pen, name='Right')
                
                self.plot_widgets[var] = plot_widget
                self.left_curves[var] = left_curve
                self.right_curves[var] = right_curve
                
                self.grid_layout.addWidget(plot_widget, row, col)
    
    def clear_data(self):
        """모든 데이터 초기화"""
        self.write_index = 0
        self.valid_data_count = 0
        
        # 모든 버퍼를 0으로 초기화
        for var in self.left_data_buffers:
            self.left_data_buffers[var].fill(0)
            self.right_data_buffers[var].fill(0)
        
        # 그래프도 초기화 (전체 0 데이터로)
        self.update_curves()
    
    def reset_and_start_from_position(self):
        """슬라이더 이동시 그래프를 초기화하고 처음부터 다시 시작"""
        self.clear_data()
    
    def update_plot(self, data_dict):
        """곡선 플롯 업데이트"""
        if data_dict is None:
            return
        
        left_data = data_dict.get('left', {})
        right_data = data_dict.get('right', {})
        
        # 현재 쓰기 위치에 데이터 저장
        idx = self.write_index
        
        # 각 필드의 평균값 계산하고 버퍼에 저장
        for name, addr, length, size, var in self.data_sheet:
            # 왼손 데이터
            if var in left_data:
                avg_val = np.mean(left_data[var])
                self.left_data_buffers[var][idx] = avg_val
            else:
                self.left_data_buffers[var][idx] = 0
            
            # 오른손 데이터
            if var in right_data:
                avg_val = np.mean(right_data[var])
                self.right_data_buffers[var][idx] = avg_val
            else:
                self.right_data_buffers[var][idx] = 0
        
        # 인덱스 업데이트 (circular buffer)
        self.write_index = (self.write_index + 1) % self.total_samples
        
        # 유효한 데이터 개수 업데이트 (최대 total_samples까지)
        if self.valid_data_count < self.total_samples:
            self.valid_data_count += 1
        
        # 곡선 업데이트
        self.update_curves()
    
    def get_display_data(self):
        """표시할 데이터 정렬해서 반환 (유효한 데이터만)"""
        if self.valid_data_count == 0:
            return None
        
        if self.valid_data_count < self.total_samples:
            # 아직 버퍼가 다 차지 않은 경우: 유효한 데이터만 표시
            time_display = self.time_array[:self.valid_data_count]
            
            left_data = {}
            right_data = {}
            for var in self.left_data_buffers:
                left_data[var] = self.left_data_buffers[var][:self.valid_data_count]
                right_data[var] = self.right_data_buffers[var][:self.valid_data_count]
                
        else:
            # 버퍼가 가득 찬 경우: circular buffer 순서대로 정렬
            time_display = self.time_array
            
            # write_index 이후부터 끝까지 + 처음부터 write_index까지
            indices = np.concatenate([
                np.arange(self.write_index, self.total_samples),
                np.arange(0, self.write_index)
            ])
            
            left_data = {}
            right_data = {}
            for var in self.left_data_buffers:
                left_data[var] = self.left_data_buffers[var][indices]
                right_data[var] = self.right_data_buffers[var][indices]
        
        return time_display, left_data, right_data
    
    def update_curves(self):
        """현재 버퍼 데이터로 곡선들 업데이트"""
        display_data = self.get_display_data()
        if display_data is None:
            return
        
        time_display, left_data, right_data = display_data
        
        # 각 필드의 곡선 업데이트
        for name, addr, length, size, var in self.data_sheet:
            if var in self.left_curves and var in self.right_curves:
                left_array = left_data[var]
                right_array = right_data[var]
                
                if len(left_array) > 0 and len(time_display) == len(left_array):
                    self.left_curves[var].setData(time_display, left_array)
                    self.right_curves[var].setData(time_display, right_array)

class JointDataTab(QWidget):
    """JSON 데이터에서 arm과 end-effector의 state/action 데이터를 표시하는 탭 (오실로스코프 방식)"""
    
    def __init__(self, history_len=150):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 제목 라벨
        self.title_label = QLabel("JOINT DATA (State & Action)")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: purple;")
        self.layout.addWidget(self.title_label)
        
        # 고정된 시간 윈도우 (5초)
        self.time_window = 5.0
        # 샘플링 주파수 (30Hz)
        self.sample_rate = 30.0
        # 총 샘플 수
        self.total_samples = int(self.time_window * self.sample_rate)  # 150 samples
        
        # 고정된 시간 배열 (0 ~ 5초)
        self.time_array = np.linspace(0, self.time_window, self.total_samples)
        
        # 현재 쓰기 위치 (circular buffer index)
        self.write_index = 0
        
        # 데이터 버퍼 초기화 (모두 0으로)
        # arm 데이터용 (state 7개 + action 7개)
        self.left_arm_state_buffer = np.zeros((7, self.total_samples))
        self.left_arm_action_buffer = np.zeros((7, self.total_samples))
        self.right_arm_state_buffer = np.zeros((7, self.total_samples))
        self.right_arm_action_buffer = np.zeros((7, self.total_samples))
        
        # ee 데이터용 (state 6개 + action 6개)
        self.left_ee_state_buffer = np.zeros((6, self.total_samples))
        self.left_ee_action_buffer = np.zeros((6, self.total_samples))
        self.right_ee_state_buffer = np.zeros((6, self.total_samples))
        self.right_ee_action_buffer = np.zeros((6, self.total_samples))
        
        # 유효한 데이터 개수 추적 (처음에는 0부터 시작)
        self.valid_data_count = 0
        
        self.create_plots()
    
    def create_plots(self):
        """2x2 그리드로 플롯 생성"""
        self.grid_layout = QGridLayout()
        self.layout.addLayout(self.grid_layout)
        
        # 플롯 위젯들과 곡선들을 저장할 딕셔너리
        self.plot_widgets = {}
        self.state_curves = {}
        self.action_curves = {}
        
        # 2x2 레이아웃 정의
        plot_configs = [
            ('left_arm', 0, 0, 'Left Arm (7 DOF)', 'blue'),
            ('right_arm', 0, 1, 'Right Arm (7 DOF)', 'red'),
            ('left_ee', 1, 0, 'Left End-Effector (6 DOF)', 'darkblue'),
            ('right_ee', 1, 1, 'Right End-Effector (6 DOF)', 'darkred')
        ]
        
        for key, row, col, title, base_color in plot_configs:
            # 플롯 위젯 생성
            plot_widget = pg.PlotWidget()
            plot_widget.setTitle(title)
            plot_widget.setLabel('left', 'Joint Value')
            plot_widget.setLabel('bottom', 'Time (s)')
            plot_widget.addLegend()
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # x축 범위 완전 고정 (절대 변하지 않음)
            plot_widget.setXRange(0, self.time_window, padding=0)
            plot_widget.enableAutoRange(axis='x', enable=False)  # x축 자동 스케일링 비활성화
            plot_widget.setMouseEnabled(x=False, y=True)  # x축 마우스 줌/팬 비활성화
            
            # ViewBox의 x축 범위도 강제로 고정
            view_box = plot_widget.getViewBox()
            view_box.setLimits(xMin=0, xMax=self.time_window)
            
            self.plot_widgets[key] = plot_widget
            self.state_curves[key] = []
            self.action_curves[key] = []
            
            # arm은 7개, ee는 6개 조인트
            num_joints = 7 if 'arm' in key else 6
            
            # state와 action 곡선들 생성
            for i in range(num_joints):
                # state 곡선 (실선)
                state_pen = pg.mkPen(color=base_color, width=2, style=QtCore.Qt.SolidLine)
                state_curve = plot_widget.plot(pen=state_pen, name=f'State_{i}')
                self.state_curves[key].append(state_curve)
                
                # action 곡선 (점선)
                action_pen = pg.mkPen(color=base_color, width=2, style=QtCore.Qt.DashLine)
                action_curve = plot_widget.plot(pen=action_pen, name=f'Action_{i}')
                self.action_curves[key].append(action_curve)
            
            # 그리드에 추가
            self.grid_layout.addWidget(plot_widget, row, col)
    
    def clear_data(self):
        """모든 데이터 초기화"""
        self.write_index = 0
        self.valid_data_count = 0
        
        # 모든 버퍼를 0으로 초기화
        self.left_arm_state_buffer.fill(0)
        self.left_arm_action_buffer.fill(0)
        self.right_arm_state_buffer.fill(0)
        self.right_arm_action_buffer.fill(0)
        
        self.left_ee_state_buffer.fill(0)
        self.left_ee_action_buffer.fill(0)
        self.right_ee_state_buffer.fill(0)
        self.right_ee_action_buffer.fill(0)
        
        # 그래프도 초기화 (전체 0 데이터로)
        self.update_curves()
    
    def reset_and_start_from_position(self):
        """슬라이더 이동시 그래프를 초기화하고 처음부터 다시 시작"""
        self.clear_data()
    
    def update_plot(self, json_data_dict):
        """JSON 데이터를 이용해 플롯 업데이트"""
        if json_data_dict is None or 'data' not in json_data_dict:
            return
        
        data = json_data_dict['data']
        
        # states와 actions 데이터 추출
        states = data.get('states', {})
        actions = data.get('actions', {})
        
        # 현재 쓰기 위치에 데이터 저장
        idx = self.write_index
        
        # Left Arm 데이터 처리
        if 'left_arm' in states and 'qpos' in states['left_arm']:
            left_arm_state = states['left_arm']['qpos']
            for i, val in enumerate(left_arm_state[:7]):
                self.left_arm_state_buffer[i, idx] = val
        else:
            for i in range(7):
                self.left_arm_state_buffer[i, idx] = 0
        
        if 'left_arm' in actions and 'qpos' in actions['left_arm']:
            left_arm_action = actions['left_arm']['qpos']
            for i, val in enumerate(left_arm_action[:7]):
                self.left_arm_action_buffer[i, idx] = val
        else:
            for i in range(7):
                self.left_arm_action_buffer[i, idx] = 0
        
        # Right Arm 데이터 처리
        if 'right_arm' in states and 'qpos' in states['right_arm']:
            right_arm_state = states['right_arm']['qpos']
            for i, val in enumerate(right_arm_state[:7]):
                self.right_arm_state_buffer[i, idx] = val
        else:
            for i in range(7):
                self.right_arm_state_buffer[i, idx] = 0
        
        if 'right_arm' in actions and 'qpos' in actions['right_arm']:
            right_arm_action = actions['right_arm']['qpos']
            for i, val in enumerate(right_arm_action[:7]):
                self.right_arm_action_buffer[i, idx] = val
        else:
            for i in range(7):
                self.right_arm_action_buffer[i, idx] = 0
        
        # Left EE 데이터 처리
        if 'left_ee' in states and 'qpos' in states['left_ee']:
            left_ee_state = states['left_ee']['qpos']
            for i, val in enumerate(left_ee_state[:6]):
                self.left_ee_state_buffer[i, idx] = val
        else:
            for i in range(6):
                self.left_ee_state_buffer[i, idx] = 0
        
        if 'left_ee' in actions and 'qpos' in actions['left_ee']:
            left_ee_action = actions['left_ee']['qpos']
            for i, val in enumerate(left_ee_action[:6]):
                self.left_ee_action_buffer[i, idx] = val
        else:
            for i in range(6):
                self.left_ee_action_buffer[i, idx] = 0
        
        # Right EE 데이터 처리
        if 'right_ee' in states and 'qpos' in states['right_ee']:
            right_ee_state = states['right_ee']['qpos']
            for i, val in enumerate(right_ee_state[:6]):
                self.right_ee_state_buffer[i, idx] = val
        else:
            for i in range(6):
                self.right_ee_state_buffer[i, idx] = 0
        
        if 'right_ee' in actions and 'qpos' in actions['right_ee']:
            right_ee_action = actions['right_ee']['qpos']
            for i, val in enumerate(right_ee_action[:6]):
                self.right_ee_action_buffer[i, idx] = val
        else:
            for i in range(6):
                self.right_ee_action_buffer[i, idx] = 0
        
        # 인덱스 업데이트 (circular buffer)
        self.write_index = (self.write_index + 1) % self.total_samples
        
        # 유효한 데이터 개수 업데이트 (최대 total_samples까지)
        if self.valid_data_count < self.total_samples:
            self.valid_data_count += 1
        
        # 곡선 업데이트
        self.update_curves()
    
    def get_display_data(self):
        """표시할 데이터 정렬해서 반환 (유효한 데이터만)"""
        if self.valid_data_count == 0:
            return None
        
        if self.valid_data_count < self.total_samples:
            # 아직 버퍼가 다 차지 않은 경우: 유효한 데이터만 표시
            time_display = self.time_array[:self.valid_data_count]
            
            left_arm_state = self.left_arm_state_buffer[:, :self.valid_data_count]
            left_arm_action = self.left_arm_action_buffer[:, :self.valid_data_count]
            right_arm_state = self.right_arm_state_buffer[:, :self.valid_data_count]
            right_arm_action = self.right_arm_action_buffer[:, :self.valid_data_count]
            left_ee_state = self.left_ee_state_buffer[:, :self.valid_data_count]
            left_ee_action = self.left_ee_action_buffer[:, :self.valid_data_count]
            right_ee_state = self.right_ee_state_buffer[:, :self.valid_data_count]
            right_ee_action = self.right_ee_action_buffer[:, :self.valid_data_count]
            
        else:
            # 버퍼가 가득 찬 경우: circular buffer 순서대로 정렬
            time_display = self.time_array
            
            # write_index 이후부터 끝까지 + 처음부터 write_index까지
            indices = np.concatenate([
                np.arange(self.write_index, self.total_samples),
                np.arange(0, self.write_index)
            ])
            
            left_arm_state = self.left_arm_state_buffer[:, indices]
            left_arm_action = self.left_arm_action_buffer[:, indices]
            right_arm_state = self.right_arm_state_buffer[:, indices]
            right_arm_action = self.right_arm_action_buffer[:, indices]
            left_ee_state = self.left_ee_state_buffer[:, indices]
            left_ee_action = self.left_ee_action_buffer[:, indices]
            right_ee_state = self.right_ee_state_buffer[:, indices]
            right_ee_action = self.right_ee_action_buffer[:, indices]
        
        return (time_display, left_arm_state, left_arm_action, right_arm_state,
                right_arm_action, left_ee_state, left_ee_action, right_ee_state, right_ee_action)
    
    def update_curves(self):
        """현재 버퍼 데이터로 곡선들 업데이트"""
        display_data = self.get_display_data()
        if display_data is None:
            return
        
        (time_display, left_arm_state, left_arm_action, right_arm_state,
         right_arm_action, left_ee_state, left_ee_action, right_ee_state, right_ee_action) = display_data
        
        # Left Arm 곡선 업데이트
        for i in range(7):
            self.state_curves['left_arm'][i].setData(time_display, left_arm_state[i])
            self.action_curves['left_arm'][i].setData(time_display, left_arm_action[i])
        
        # Right Arm 곡선 업데이트
        for i in range(7):
            self.state_curves['right_arm'][i].setData(time_display, right_arm_state[i])
            self.action_curves['right_arm'][i].setData(time_display, right_arm_action[i])
        
        # Left EE 곡선 업데이트
        for i in range(6):
            self.state_curves['left_ee'][i].setData(time_display, left_ee_state[i])
            self.action_curves['left_ee'][i].setData(time_display, left_ee_action[i])
        
        # Right EE 곡선 업데이트
        for i in range(6):
            self.state_curves['right_ee'][i].setData(time_display, right_ee_state[i])
            self.action_curves['right_ee'][i].setData(time_display, right_ee_action[i])

class NPYViewerMainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unitree G1 Teleoperation DataSet Viewer")
        self.setGeometry(100, 100, 1400, 1000)
        
        self.data_handler = None
        self.current_data = None
        
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        """UI 설정"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # 컨트롤 패널
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 탭 위젯
        self.tabs = QTabWidget()
        # 탭 텍스트 크기 설정 (Select Folder 버튼보다 조금 작게)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 8px 20px;
                margin-right: 2px;
                font-size: 16px;
                min-width: 200px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom-color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #e0e0e0;
            }
        """)
        
        # 왼손, 오른손, 카메라, 조인트용 별도 탭 생성
        self.left_image_tab = TactileImageTab(data_sheet, hand='left')
        self.right_image_tab = TactileImageTab(data_sheet, hand='right')
        self.camera_tab = CameraImageTab()
        self.curve_tab = TactileCurveTab(data_sheet)
        self.joint_tab = JointDataTab()
        
        # 탭 순서 및 이름 변경
        self.tabs.addTab(self.camera_tab, "Head & Wrist Images")
        self.tabs.addTab(self.joint_tab, "Arm & Hand Joint Data")
        self.tabs.addTab(self.left_image_tab, "Left Hand Tactile Data")
        self.tabs.addTab(self.right_image_tab, "Right Hand Tactile Data")
        self.tabs.addTab(self.curve_tab, "Hand Tactile Curves")
        
        layout.addWidget(self.tabs)
    
    def create_control_panel(self):
        """컨트롤 패널 생성"""
        panel = QWidget()
        layout = QHBoxLayout()
        panel.setLayout(layout)
        
        # 폴더 선택 버튼 (폴더 아이콘 추가)
        self.folder_btn = QPushButton("📁 Select Folder")
        self.folder_btn.clicked.connect(self.select_folder)
        self.folder_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 24px;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            QPushButton:hover {
                background-color: #e8f5e8;
            }
        """)
        layout.addWidget(self.folder_btn)
        
        # 재생 컨트롤
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 24px;
                border: 2px solid #2196F3;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            QPushButton:hover {
                background-color: #e3f2fd;
            }
            QPushButton:disabled {
                color: #888;
                border-color: #ccc;
                background-color: #f0f0f0;
            }
        """)
        layout.addWidget(self.play_btn)
        
        self.prev_btn = QPushButton("⏮️ Prev")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.setEnabled(False)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 24px;
                border: 2px solid #FF9800;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            QPushButton:hover {
                background-color: #fff3e0;
            }
            QPushButton:disabled {
                color: #888;
                border-color: #ccc;
                background-color: #f0f0f0;
            }
        """)
        layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("⏭️ Next")
        self.next_btn.clicked.connect(self.next_frame)
        self.next_btn.setEnabled(False)
        self.next_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 12px 24px;
                border: 2px solid #FF9800;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            QPushButton:hover {
                background-color: #fff3e0;
            }
            QPushButton:disabled {
                color: #888;
                border-color: #ccc;
                background-color: #f0f0f0;
            }
        """)
        layout.addWidget(self.next_btn)
        
        # 속도 조절
        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(speed_label)

        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(1, 120)
        self.speed_spinbox.setValue(30)
        self.speed_spinbox.setSuffix(" fps")
        self.speed_spinbox.setStyleSheet("font-size: 16px;")
        self.speed_spinbox.valueChanged.connect(self.update_timer_interval)
        layout.addWidget(self.speed_spinbox)

        # 프레임 슬라이더
        frame_label = QLabel("Frame:")
        frame_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(frame_label)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.frame_slider.setEnabled(False)
        layout.addWidget(self.frame_slider)

        # 진행률 표시
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.progress_bar)

        # 정보 라벨
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.info_label)
        
        return panel
    
    def setup_timer(self):
        """타이머 설정"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False
        self.update_timer_interval()
    
    def update_timer_interval(self):
        """타이머 간격 업데이트"""
        fps = self.speed_spinbox.value()
        interval = 1000 // fps  # ms
        self.timer.setInterval(interval)
    
    def select_folder(self):
        """폴더 선택 - tactiles, colors, data.json을 포함하는 상위 폴더"""
        folder = QFileDialog.getExistingDirectory(self, "Select Episode Directory (containing tactiles, colors folders and data.json)")
        if folder:
            self.data_handler = NPYDataHandler(folder)
            total_frames = self.data_handler.get_total_frames()
            
            if total_frames > 0:
                self.frame_slider.setRange(0, total_frames - 1)
                self.frame_slider.setValue(0)
                self.frame_slider.setEnabled(True)
                self.play_btn.setEnabled(True)
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
                
                self.progress_bar.setRange(0, total_frames - 1)
                
                # 조인트 데이터 탭 초기화 (처음 로드시)
                self.joint_tab.reset_and_start_from_position()

                # Tactile Curve 탭 초기화 추가 (처음 로드시)
                self.curve_tab.reset_and_start_from_position()
                
                self.update_display()
                
                print(f"Loaded {total_frames} frames from {folder}")
            else:
                self.info_label.setText("No valid data files found in selected folder")
                print("Make sure the folder contains 'tactiles', 'colors' subdirectories and 'data.json' file")
    
    def toggle_playback(self):
        """재생/정지 토글"""
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("▶️ Play")
            self.is_playing = False
        else:
            self.timer.start()
            self.play_btn.setText("⏸️ Pause")
            self.is_playing = True
    
    def update_frame(self):
        """프레임 업데이트 (자동 재생용)"""
        if self.data_handler:
            if not self.data_handler.next_frame():
                # 끝에 도달하면 처음으로
                self.data_handler.set_frame(0)
                # ★ 새로 추가: 처음으로 돌아갈 때 히스토리 재구성
                
            self.update_display()
    
    def prev_frame(self):
        """이전 프레임"""
        if self.data_handler:
            # 조인트 데이터 탭 그래프 초기화
            self.joint_tab.reset_and_start_from_position()
            # Tactile Curve 탭 그래프 초기화 추가
            self.curve_tab.reset_and_start_from_position()

            self.data_handler.prev_frame()
            self.update_display()

    def next_frame(self):
        """다음 프레임"""
        if self.data_handler:
            # 조인트 데이터 탭 그래프 초기화
            self.joint_tab.reset_and_start_from_position()
            # Tactile Curve 탭 그래프 초기화 추가
            self.curve_tab.reset_and_start_from_position()

            self.data_handler.next_frame()
            self.update_display()
    
    def slider_changed(self):
        """슬라이더 변경"""
        if self.data_handler:
            frame_index = self.frame_slider.value()
            
            # 조인트 데이터 탭 그래프 초기화 (처음부터 다시 그리기)
            self.joint_tab.reset_and_start_from_position()
            # Tactile Curve 탭 그래프 초기화 추가
            self.curve_tab.reset_and_start_from_position()
            self.data_handler.set_frame(frame_index)
            self.update_display()

    
    def update_display(self):
        """디스플레이 업데이트 - tactile, 카메라 이미지, JSON 데이터 모두"""
        if not self.data_handler:
            return
        
        # Tactile 데이터 읽기
        tactile_data = self.data_handler.read_current_frame()
        # 카메라 이미지 읽기
        image_data = self.data_handler.read_current_images()
        # JSON 데이터 읽기
        json_data = self.data_handler.read_current_json_data()
        
        if tactile_data:
            # 좌우 tactile 데이터 분리
            left_data = tactile_data.get('left', {})
            right_data = tactile_data.get('right', {})
            
            # 각 tactile 탭에게 해당 손의 데이터만 전달
            left_data_only = {
                'left': left_data,
                'index': tactile_data['index'], 
                'time': tactile_data['time']
            }
            
            right_data_only = {
                'right': right_data,
                'index': tactile_data['index'], 
                'time': tactile_data['time']
            }
            
            # Tactile 탭 업데이트
            self.left_image_tab.update_plot(left_data_only)
            self.right_image_tab.update_plot(right_data_only)
            self.curve_tab.update_plot(tactile_data)
            
            # UI 정보 업데이트
            current_frame = self.data_handler.current_index
            total_frames = self.data_handler.get_total_frames()
            current_time = tactile_data['time']
            
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(current_frame)
            self.frame_slider.blockSignals(False)
            
            self.progress_bar.setValue(current_frame)
            
            self.info_label.setText(
                f"Frame: {current_frame}/{total_frames-1} | "
                f"Time: {current_time:.2f}s | "
                f"File: {tactile_data['index']:06d}"
            )
        
        # 카메라 이미지 업데이트
        if image_data:
            self.camera_tab.update_images(image_data)
        
        # 조인트 데이터 업데이트
        if json_data:
            self.joint_tab.update_plot(json_data)

def main():
    # Qt 환경 변수 설정 (OpenCV 충돌 방지)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
        del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
    
    app = QApplication(sys.argv)
    
    # Qt 플랫폼 플러그인 강제 설정
    app.setAttribute(Qt.AA_X11InitThreads)
    
    window = NPYViewerMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()