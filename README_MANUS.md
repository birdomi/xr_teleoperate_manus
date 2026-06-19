# Manus + Vive Tracker 텔레오퍼레이션

이 문서는 `teleop/teleop_hand_and_arm_manus.py`의 사용법과 내부 데이터 흐름을
설명합니다. 이 스크립트는 Vive 트래커로 양팔을 움직이고, Manus 글러브로 손을
리타게팅하는 텔레오퍼레이션 진입점입니다.

## 스크립트 역할

`teleop_hand_and_arm_manus.py`는 다음 입력을 동시에 읽습니다.

- `libsurvive_ros2`가 publish하는 Vive 트래커 TF
- Manus 글러브 ROS2 topic
- 선택 사항: 머리 Vive 트래커를 이용한 시뮬레이터 카메라 제어

읽은 입력은 기존 `TeleData` 형식으로 변환됩니다.

- `left_arm_pose`, `right_arm_pose`: 로봇 팔 IK에 들어가는 EE 목표 pose
- `left_hand_pos`, `right_hand_pos`: 손 리타게팅에 쓰이는 25개 wrist-local 점
- 손 pose는 shared array에 기록되고, 선택한 hand controller가 별도 process/thread에서 읽습니다.

## 설치 참고 링크

전체 설치 흐름은 MANUS 공식 문서와 `libsurvive_ros2` 문서를 기준으로 진행합니다.

- MANUS ROS2 설치 및 사용:
  https://docs.manus-meta.com/latest/Plugins/SDK/ROS2/getting%20started/#installation
- MANUS SDK Linux 설치:
  https://docs.manus-meta.com/latest/Plugins/SDK/Linux/
- MANUS Core 3 SDK 다운로드:
  https://docs.manus-meta.com/latest/Plugins/SDK/getting%20started/
- Vive Tracker ROS2 driver:
  https://github.com/asymingt/libsurvive_ros2

## 환경 준비

### MANUS HAND

MANUS 쪽은 공식 ROS2 guide의 흐름을 따르면 됩니다.

1. MANUS SDK Linux 설치 문서를 따라 Linux SDK 환경을 준비합니다.
2. MANUS SDK getting started 문서에서 `MANUS Core 3 SDK (including ROS2 Package)`를 다운로드합니다.
3. `manus_ws` workspace를 만들고 `src` 폴더를 생성합니다.
4. 다운로드한 SDK 안의 `ROS2` 폴더 내용을 `manus_ws/src` 안으로 복사합니다.
5. `colcon build`로 workspace를 빌드합니다.

예시:

```bash
mkdir -p ~/projects/manus_ws/src
cd ~/projects/manus_ws
# SDK 안의 ROS2 폴더를 ~/projects/manus_ws/src 안으로 복사
colcon build
```

빌드 후 MANUS publisher는 별도 터미널에서 실행합니다.

```bash
cd ~/projects/manus_ws
source install/setup.bash
ros2 run manus_ros2 manus_data_publisher
```

### Vive Tracker

Vive tracker는 `libsurvive_ros2`를 사용합니다. 해당 repository의
`Native build and test (not recommended)` 절차를 따라 설치합니다.

설치 후 별도 터미널에서 다음을 실행합니다.

```bash
cd ~/projects/vive_ws
source install/setup.bash
ros2 launch libsurvive_ros2 libsurvive_ros2.launch.py rosbridge:=true
```

현재 착용 기준:

- 왼손 tracker: `s2`
- 오른손 tracker: `s3`
- 양쪽 tracker의 초록색 불빛이 서로 마주보게 착용

`libsurvive_ros2`의 TF frame 이름이 `s2`, `s3`로 올라오면 teleop 실행 시
`--left-tracker-name s2 --right-tracker-name s3`를 명시합니다. 코드의 기본 tracker
이름과 실제 TF frame 이름이 같다면 생략해도 됩니다.

## 실행 순서

아래 순서대로 터미널을 4개 열어 실행합니다.

### 터미널 1: MANUS publisher

```bash
cd ~/projects/manus_ws
source install/setup.bash
ros2 run manus_ros2 manus_data_publisher
```

### 터미널 2: Vive tracker publisher

```bash
cd ~/projects/vive_ws
source install/setup.bash
ros2 launch libsurvive_ros2 libsurvive_ros2.launch.py rosbridge:=true
```

### 터미널 3: Unitree simulation

시뮬레이터를 실행하기 전에 CycloneDDS 경로가 필요한 환경이면 먼저 export합니다.

```bash
export CYCLONEDDS_HOME=/home/goodman/projects/vive_ws_temp/unitree_sdk2_python/cyclonedds/install
export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH
```

그 다음 시뮬레이션을 실행합니다.

```bash
conda activate unitree_sim_env
cd ~/projects/unitree_sim_isaaclab
python sim_main.py \
  --device cpu \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --enable_inspire_dds \
  --robot_type g129
```

### 터미널 4: teleop 코드

기본 실행:

```bash
conda activate tv
cd ~/projects/xr_teleoperate/teleop
python teleop_hand_and_arm_manus.py --ee inspire1 --sim
```

Vive tracker TF frame이 `s2`, `s3`로 올라오는 세팅에서는 다음처럼 명시합니다.

```bash
python teleop_hand_and_arm_manus.py \
  --ee inspire1 \
  --sim \
  --left-tracker-name s2 \
  --right-tracker-name s3
```

teleop 프로그램이 `STANDBY`에 들어가면 양손 tracker 회전을 sim EE 회전과 맞춘 뒤
터미널에서 `r`을 누릅니다. 프로그램은 3초 기다린 뒤 sync 검사를 수행하고,
회전 차이가 threshold 안에 있으면 `ACTIVE`로 진입합니다.

## 데이터 흐름

1. `LibsurviveTFReader`
   - 왼손, 오른손, 머리 트래커의 TF transform을 읽습니다.
   - TF의 translation + quaternion을 4x4 pose matrix로 변환합니다.
   - 짧은 TF drop이 있어도 바로 target이 0이 되지 않도록 마지막 정상 pose를 잠시 유지합니다.

2. `ManusHandReader`
   - 설정된 Manus ROS2 topic을 subscribe합니다.
   - `msg.side` 값으로 왼손/오른손을 구분합니다.
   - `raw_nodes[0..24]`를 25개 손 관절 점으로 읽습니다.
   - `raw_nodes[0]`의 pose를 wrist frame으로 사용합니다.

3. `ViveManusTeleopWrapper`
   - Vive/libsurvive pose를 로봇 world 좌표계로 변환합니다.
   - 트래커 device frame을 Unitree EE frame에 맞게 보정합니다.
   - `r` sync 때 저장한 calibration pose를 기준으로 트래커의 상대 이동을 계산합니다.
   - Manus world-space hand node를 wrist-local URDF 좌표로 변환합니다.

4. Main loop
   - 매 frame `tv_wrapper.get_tele_data()`를 호출합니다.
   - `dex3`, `inspire1`, `brainco`를 사용할 때 Manus hand position을
     `left_hand_pos_array`, `right_hand_pos_array`에 씁니다.
   - 팔 IK는 tracker/sim sync가 성공한 뒤에만 동작합니다.

## 좌표계

기본 Vive 입력 frame은 `libsurvive`입니다.

- libsurvive: `X=right`, `Y=forward`, `Z=up`
- robot world: `X=forward`, `Y=left`, `Z=up`

코드에서는 다음 방식으로 libsurvive pose를 robot world pose로 바꿉니다.

```python
T_robot_tracker = _T_LIBSURVIVE_TO_ROBOT @ T_libsurvive_tracker @ _T_ROBOT_TO_LIBSURVIVE
```

그 다음 트래커 device frame을 Unitree EE frame에 맞추기 위해 다음 보정을 적용합니다.

```python
_T_WRIST_CORR_LEFT
_T_WRIST_CORR_RIGHT
```

왼손 트래커는 장착 방향 때문에 local X축이 원하는 EE frame 기준으로 뒤쪽을 보고
있습니다. 그래서 왼손에는 추가로 `_LEFT_X_FORWARD_FIX`를 적용합니다.

## `r` 키로 시작 sync

프로그램은 처음에 `STANDBY` 상태로 시작합니다. 터미널에서 `r`을 누르면 sync를
요청합니다.

동작 순서:

1. `r` 입력 후 3초 기다립니다.
2. 현재 Vive tracker pose와 현재 sim/robot EE pose를 읽습니다.
3. tracker 회전과 sim EE 회전 차이를 비교합니다.
4. 회전 차이가 너무 크면 sync 실패로 처리하고 `STANDBY`에 남습니다.
5. 회전 차이가 허용 범위 안이면 현재 tracker pose를 현재 sim EE pose에
   calibration하고 `ACTIVE`로 진입합니다.

위치 차이는 출력만 하고 sync 실패 조건에는 사용하지 않습니다. 팔 움직임은 sync
이후 tracker의 상대 이동으로 계산하기 때문입니다.

기본 회전 threshold는 45도입니다.

```bash
--sync-max-rotation-error-deg 45.0
```

`--sync-max-position-error` 옵션은 호환성을 위해 남아 있지만, 현재 sync 실패
판정에는 사용하지 않습니다.

## 키 입력

- `r`: 3초 기다린 뒤 tracker/sim 회전 sync를 검사하고, 성공하면 `ACTIVE` 진입
- `c`: wrist tracker와 head camera neutral 재캘리브레이션
- `s`: `--record` 사용 시 녹화 시작/종료
- `a`: `--sim` 사용 시 simulator scene reset
- `q`: 종료

## 추가 실행 예시

G1 팔, BrainCo 손, Manus 글러브, Vive 트래커, simulator mode를 사용하는 예시입니다.

```bash
cd teleop
python teleop_hand_and_arm_manus.py \
  --arm G1_29 \
  --ee brainco \
  --sim \
  --left-tracker-name LHR-2F1F34FC \
  --right-tracker-name LHR-6711118F \
  --head-tracker-name LHR-501D76A5 \
  --manus-topics manus_glove_0 manus_glove_1 \
  --manus-msg-type manus_ros2_msgs/msg/ManusGlove
```

트래커 frame 이름이 다르면 `libsurvive_ros2`가 publish하는 TF frame을 확인한 뒤
다음 옵션을 실제 serial 이름으로 바꿔 실행합니다.

- `--left-tracker-name`
- `--right-tracker-name`
- `--head-tracker-name`

## 주요 옵션

- `--libsurvive-tracking-frame`
  - 기본값: `libsurvive_world`
  - 트래커 TF를 lookup할 때 기준이 되는 parent frame입니다.

- `--vive-input-frame`
  - 기본값: `libsurvive`
  - 입력 트래커 pose의 좌표계 convention을 지정합니다.
  - 선택지: `libsurvive`, `openxr`, `robot`, `waist`

- `--vive-position-scale`
  - 기본값: `1.0`
  - tracker translation delta를 팔 목표 pose에 적용할 때 스케일합니다.

- `--no-unitree-arm-orientation-fix`
  - wrist axis correction을 끕니다.
  - 일반적으로는 끄지 않는 것을 권장합니다.

- `--enable-head-camera-dds`
  - 머리 트래커 회전을 simulator camera DDS topic으로 publish합니다.

## 문제 해결

### `r`을 눌렀는데 sync가 실패하는 경우

- `[SYNC ALARM]` 출력 내용을 확인합니다.
- 왼손/오른손 각각의 `rot_error` 값을 봅니다.
- 트래커 장착 방향이 기대한 방향과 맞는지 확인합니다.
- 장착 방향 차이를 의도적으로 허용해야 하는 경우에만
  `--sync-max-rotation-error-deg` 값을 키웁니다.

### 손 리타게팅이 안 되는 경우

- Manus topic 두 개가 publish되고 있는지 확인합니다.
- `msg.side` 값이 `left` 또는 `right`인지 확인합니다.
- `raw_nodes`가 25개 node를 포함하는지 확인합니다.
- `raw_nodes[0]`에 유효한 wrist pose가 들어오는지 확인합니다.

### 팔 움직임이 갑자기 튀는 경우

- `q`로 안전하게 종료합니다.
- sim 또는 로봇 팔 pose를 다시 준비 자세로 맞춥니다.
- tracker 회전이 현재 sim EE 회전과 충분히 가까운 상태에서 다시 `r`을 누릅니다.
