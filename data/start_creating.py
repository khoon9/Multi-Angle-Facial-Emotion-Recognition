import subprocess
import os

# Blender 실행 경로와 스크립트 파일 경로
# python_script = "render_obj_to_png_30_30degree.py"  # Blender Python 스크립트 파일 경로
python_script = "render_obj_to_png_60_40degree.py"  # Blender Python 스크립트 파일 경로

# OBJ 파일 경로 목록
image_dir = 'input'
image_files_A = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('A.obj'))] # 무표정
image_files_B = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('B.obj'))] # 행복
# image_files_C = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('C.obj'))] # 슬픔
# image_files_D = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('D.obj'))] # 두려움
image_files_E = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('E.obj'))] # 화남
image_files_F = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('F.obj'))] # 놀람
# image_files_G = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('G.obj'))] # 역겨움

# 예상치 못한 렌더링 상황을 방지하기 위해, 각 렌더링을 구분지어 순차적으로 호출하는 함수
def start_creating_by_cmd(blender_executable,path_list,file_num,increment):
    for obj_path in path_list:
        command = [
            blender_executable,
            "--background",
            "--python", python_script,
            "--", obj_path, str(file_num), './output'
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)
        file_num = file_num + increment

# 렌더링할 범위를 조절
# image_files_A = image_files_A[250:]
# image_files_B = image_files_B[250:]
# image_files_E = image_files_E[250:]
# image_files_F = image_files_F[250:]

# image_files_A = image_files_A[250:]
# image_files_B = image_files_B[250:]
# image_files_E = image_files_E[:2]
# image_files_F = image_files_F[:3]

# 다중 터미널에서 각 분류군을 병렬로 실행하기 위해 구성한 코드
# start_creating_by_cmd("blender",image_files_A,0*477*13*9+250*13*9,13*9)
# start_creating_by_cmd("blender_a",image_files_B,1*477*13*9+250*13*9,13*9)
# start_creating_by_cmd("blender_b",image_files_E,2*477*13*9,13*9)
start_creating_by_cmd("blender_c",image_files_F,3*477*13*9,13*9)
print("All commands executed.")
