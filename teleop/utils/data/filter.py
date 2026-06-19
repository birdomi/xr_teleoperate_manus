import os
import cv2
from pathlib import Path
import argparse

def process_target_jpgs(folder_path: str):
    """
    선택한 폴더 내에서 '_1.jpg', '_2.jpg', '_3.jpg'로 끝나는 
    파일을 찾아 삭제하고, '_0.jpg' 파일은 사이즈를 줄여(용량 축소) 덮어쓰는 함수입니다.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: 유효하지 않은 폴더 경로입니다: {folder_path}")
        return

    # 1. 삭제 및 처리(삭제/보존) 파일 패턴 설정
    delete_patterns = ["*_1.jpg", "*_2.jpg", "*_3.jpg"]
    deleted_count = 0

    print(f"[{folder_path}] 폴더 내 파일 삭제 시작...")
    
    for pattern in delete_patterns:
        # 지정된 패턴과 일치하는 모든 하위 폴더의 파일 탐색 (colors 폴더 내의 파일만)
        for file_path in folder.rglob(f"colors/{pattern}"):
            if file_path.is_file():
                try:
                    file_path.unlink()  # 파일 삭제
                    print(f"삭제됨: {file_path.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"삭제 실패 ({file_path.name}): {e}")
    
    print(f"완료! 총 {deleted_count}개의 파일을 삭제했습니다.")

    # 2. _0.jpg 이미지 축소 처리
    resize_pattern = "*_0.jpg"
    resized_count = 0

    print(f"[{folder_path}] 폴더 내 '_0.jpg' 파일 용량 축소 시작...")
    
    for file_path in folder.rglob(f"colors/{resize_pattern}"):
        if file_path.is_file():
            try:
                # 이미지 읽기
                img = cv2.imread(str(file_path))
                if img is not None:
                    # 절반 크기로 줄이기 (해상도 감소)
                    resized_img = cv2.resize(img, (0, 0), fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)
                    
                    # 압축률 설정하여 덮어쓰기 (JPEG 품질 85, 용량 축소)
                    cv2.imwrite(str(file_path), resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    print(f"축소됨: {file_path.name}")
                    resized_count += 1
                else:
                    print(f"이미지 읽기 실패: {file_path.name}")
            except Exception as e:
                print(f"축소 실패 ({file_path.name}): {e}")

    print(f"완료! 총 {resized_count}개의 이미지를 축소했습니다.")


if __name__ == "__main__":
    # 터미널에서 실행할 때 폴더 경로를 인자로 받을 수 있도록 설정
    parser = argparse.ArgumentParser(description="특정 패턴(_1.jpg, _2.jpg, _3.jpg)으로 끝나는 파일을 삭제하고 '_0.jpg'는 해상도와 용량을 축소합니다.")
    parser.add_argument("-d", "--dir", type=str, help="처리할 파일이 있는 폴더 경로")
    
    args = parser.parse_args()
    
    target_dir = args.dir
    if not target_dir:
        # 인자로 경로가 주어지지 않은 경우 사용자에게 직접 입력 받음
        try:
            target_dir = input("작업을 수행할 폴더 경로를 입력하세요: ").strip()
        except EOFError:
            target_dir = ""

    if target_dir:
        process_target_jpgs(target_dir)
    else:
        print("폴더 경로가 입력되지 않았습니다.")
