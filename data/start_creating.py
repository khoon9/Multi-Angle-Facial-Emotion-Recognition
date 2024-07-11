import subprocess
import os

# Blender 실행 경로와 스크립트 파일 경로
blender_executable = "blender"  # Blender 실행 파일 경로
python_script = "render_obj_to_png.py"  # Blender Python 스크립트 파일 경로

# OBJ 파일 경로 목록
image_dir = 'input'
image_files_A = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('A.obj'))] # 무표정
image_files_B = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('B.obj'))] # 행복
# image_files_C = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('C.obj'))] # 슬픔
# image_files_D = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('D.obj'))] # 두려움
image_files_E = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('E.obj'))] # 화남
image_files_F = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('F.obj'))] # 놀람
# image_files_G = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('G.obj'))] # 역겨움

# 초기 파일 번호
file_num = 0
increment = 49
def start_creating_by_cmd(path_list):
    global file_num
    global increment
    for obj_path in path_list:
        command = [
            blender_executable,
            "--background",
            "--python", python_script,
            "--", obj_path, str(file_num)
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)
        file_num = file_num + increment

start_creating_by_cmd(image_files_A)
start_creating_by_cmd(image_files_B)
start_creating_by_cmd(image_files_E)
start_creating_by_cmd(image_files_F)
print("All commands executed.")
