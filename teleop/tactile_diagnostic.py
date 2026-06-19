#!/usr/bin/env python3
"""
Tactile 데이터 수신 진단 스크립트
DDS 연결과 tactile 데이터 수신을 테스트합니다.
"""

import sys
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from inspire_sdkpy import inspire_dds

# DDS 토픽 이름
kTopicInspireTouchLeft = "rt/inspire_hand/touch/l" 
kTopicInspireTouchRight = "rt/inspire_hand/touch/r" 

def test_tactile_connection(network_interface="enp110s0", test_duration=30):
    """
    Tactile 데이터 연결을 테스트합니다.
    
    Args:
        network_interface: 네트워크 인터페이스 이름
        test_duration: 테스트 지속 시간 (초)
    """
    print(f"=== Tactile Connection Test ===")
    print(f"Network interface: {network_interface}")
    print(f"Test duration: {test_duration} seconds")
    
    # DDS 초기화
    try:
        ChannelFactoryInitialize(0, network_interface)
        print("✓ DDS ChannelFactory initialized successfully")
    except Exception as e:
        print(f"✗ DDS initialization failed: {e}")
        return False

    # Tactile field 정의
    tactile_field_names = [
        "fingerone_tip_touch", "fingerone_top_touch", "fingerone_palm_touch",
        "fingertwo_tip_touch", "fingertwo_top_touch", "fingertwo_palm_touch",
        "fingerthree_tip_touch", "fingerthree_top_touch", "fingerthree_palm_touch",
        "fingerfour_tip_touch", "fingerfour_top_touch", "fingerfour_palm_touch",
        "fingerfive_tip_touch", "fingerfive_top_touch", "fingerfive_middle_touch", "fingerfive_palm_touch",
        "palm_touch"
    ]
    tactile_field_sizes = [9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 9, 96, 112]
    tactile_total_size = sum(tactile_field_sizes)
    
    print(f"Expected tactile fields: {len(tactile_field_names)}")
    print(f"Expected tactile total size: {tactile_total_size}")
    
    # Subscribers 초기화
    try:
        left_touch_subscriber = ChannelSubscriber(kTopicInspireTouchLeft, inspire_dds.inspire_hand_touch)
        left_touch_subscriber.Init()
        print("✓ Left tactile subscriber initialized")
        
        right_touch_subscriber = ChannelSubscriber(kTopicInspireTouchRight, inspire_dds.inspire_hand_touch)
        right_touch_subscriber.Init()
        print("✓ Right tactile subscriber initialized")
    except Exception as e:
        print(f"✗ Subscriber initialization failed: {e}")
        return False

    # 통계 변수
    left_msg_count = 0
    right_msg_count = 0
    left_data_sum = 0.0
    right_data_sum = 0.0
    left_nonzero_count = 0
    right_nonzero_count = 0
    
    start_time = time.time()
    last_report_time = start_time
    
    print("\n=== Starting tactile data monitoring ===")
    
    try:
        while time.time() - start_time < test_duration:
            current_time = time.time()
            
            # Left hand tactile 읽기
            left_touch_msg = left_touch_subscriber.Read()
            if left_touch_msg is not None:
                left_msg_count += 1
                
                # 첫 번째 메시지에서 구조 확인
                if left_msg_count == 1:
                    print(f"\n--- Left Touch Message Structure ---")
                    print(f"Message type: {type(left_touch_msg)}")
                    available_attrs = [attr for attr in dir(left_touch_msg) if not attr.startswith('_')]
                    print(f"Available attributes: {available_attrs}")
                    
                    # 각 필드 확인
                    for field_name in tactile_field_names:
                        if hasattr(left_touch_msg, field_name):
                            field_data = getattr(left_touch_msg, field_name)
                            print(f"  {field_name}: length={len(field_data)}, type={type(field_data)}")
                            if len(field_data) > 0:
                                print(f"    Sample values: {field_data[:5]}")
                        else:
                            print(f"  {field_name}: NOT FOUND")
                
                # 데이터 처리
                try:
                    total_values = []
                    for field_name in tactile_field_names:
                        if hasattr(left_touch_msg, field_name):
                            field_data = getattr(left_touch_msg, field_name)
                            total_values.extend([float(v) for v in field_data])
                    
                    if total_values:
                        left_data_sum += sum(total_values)
                        left_nonzero_count += sum(1 for v in total_values if v != 0.0)
                        
                except Exception as e:
                    print(f"Error processing left tactile data: {e}")

            # Right hand tactile 읽기
            right_touch_msg = right_touch_subscriber.Read()
            if right_touch_msg is not None:
                right_msg_count += 1
                
                # 첫 번째 메시지에서 구조 확인
                if right_msg_count == 1:
                    print(f"\n--- Right Touch Message Structure ---")
                    print(f"Message type: {type(right_touch_msg)}")
                    available_attrs = [attr for attr in dir(right_touch_msg) if not attr.startswith('_')]
                    print(f"Available attributes: {available_attrs}")
                
                # 데이터 처리
                try:
                    total_values = []
                    for field_name in tactile_field_names:
                        if hasattr(right_touch_msg, field_name):
                            field_data = getattr(right_touch_msg, field_name)
                            total_values.extend([float(v) for v in field_data])
                    
                    if total_values:
                        right_data_sum += sum(total_values)
                        right_nonzero_count += sum(1 for v in total_values if v != 0.0)
                        
                except Exception as e:
                    print(f"Error processing right tactile data: {e}")

            # 주기적으로 상태 보고
            if current_time - last_report_time >= 5.0:  # 5초마다
                elapsed = current_time - start_time
                print(f"\n--- Progress Report ({elapsed:.1f}s) ---")
                print(f"Left messages: {left_msg_count}, data sum: {left_data_sum:.2f}, nonzero: {left_nonzero_count}")
                print(f"Right messages: {right_msg_count}, data sum: {right_data_sum:.2f}, nonzero: {right_nonzero_count}")
                print(f"Message rates - L: {left_msg_count/elapsed:.1f} Hz, R: {right_msg_count/elapsed:.1f} Hz")
                last_report_time = current_time

            time.sleep(0.001)  # 1ms sleep
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    # 최종 결과
    total_time = time.time() - start_time
    print(f"\n=== Final Test Results ({total_time:.1f}s) ===")
    print(f"Left Hand:")
    print(f"  Messages received: {left_msg_count}")
    print(f"  Total data sum: {left_data_sum:.2f}")
    print(f"  Nonzero values: {left_nonzero_count}")
    print(f"  Message rate: {left_msg_count/total_time:.1f} Hz")
    
    print(f"Right Hand:")
    print(f"  Messages received: {right_msg_count}")
    print(f"  Total data sum: {right_data_sum:.2f}")
    print(f"  Nonzero values: {right_nonzero_count}")
    print(f"  Message rate: {right_msg_count/total_time:.1f} Hz")
    
    # 결과 분석
    if left_msg_count == 0 and right_msg_count == 0:
        print("\n❌ NO MESSAGES RECEIVED - Check:")
        print("  1. Network connection")
        print("  2. DDS topics are being published")
        print("  3. Network interface name")
        return False
    elif left_data_sum == 0 and right_data_sum == 0:
        print("\n⚠️  MESSAGES RECEIVED BUT ALL DATA IS ZERO - Check:")
        print("  1. Tactile sensors are active")
        print("  2. Message field names are correct")
        print("  3. Data format is as expected")
        return False
    else:
        print("\n✅ TACTILE DATA IS BEING RECEIVED SUCCESSFULLY")
        return True


def test_array_operations():
    """
    Array 연산 테스트
    """
    print(f"\n=== Array Operations Test ===")
    
    from multiprocessing import Array, Lock
    
    tactile_total_size = sum([9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 80, 9, 96, 9, 96, 112])
    
    try:
        # Array 생성 테스트
        test_array = Array('d', tactile_total_size, lock=True)
        print(f"✓ Array created successfully - Size: {len(test_array)}")
        
        # 쓰기 테스트
        with test_array.get_lock():
            for i in range(min(100, len(test_array))):
                test_array[i] = float(i)
        print(f"✓ Array write test passed")
        
        # 읽기 테스트
        with test_array.get_lock():
            data = np.array(test_array[:100])
            data_sum = np.sum(data)
        print(f"✓ Array read test passed - Sum: {data_sum}")
        
        # 전체 배열 복사 테스트
        with test_array.get_lock():
            full_data = np.array(test_array[:])
        print(f"✓ Full array copy test passed - Length: {len(full_data)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Array operations failed: {e}")
        return False


if __name__ == "__main__":
    print("Tactile Data Diagnostic Tool")
    print("="*50)
    
    # 명령행 인자 처리
    network_interface = "enp109s0"  # 기본값
    test_duration = 30  # 기본값
    
    if len(sys.argv) > 1:
        network_interface = sys.argv[1]
    if len(sys.argv) > 2:
        test_duration = int(sys.argv[2])
    
    # Array 연산 테스트
    array_test_passed = test_array_operations()
    
    if array_test_passed:
        # Tactile 연결 테스트
        connection_test_passed = test_tactile_connection(network_interface, test_duration)
        
        if connection_test_passed:
            print("\n🎉 All tests passed! Your tactile system should be working.")
        else:
            print("\n🔧 Some issues detected. Please check the recommendations above.")
    else:
        print("\n❌ Basic array operations failed. Check your Python multiprocessing setup.")
    
    print("\nDiagnostic complete.")