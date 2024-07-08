import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import os
import math

# 설정 값 직접 정의
views = 3  # 렌더링할 뷰의 수
obj_path = './input/1007M_FC_A.obj'  # 렌더링할 obj 파일의 경로
output_folder = './output'  # 출력 파일이 저장될 경로
scale = 1  # 모델에 적용될 스케일링 인수
file_format = 'PNG'  # 생성될 파일의 형식
resolution = 512  # 이미지 해상도
engine = 'CYCLES'  # 렌더링 엔진

# 렌더링 설정
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = engine
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.file_format = file_format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = resolution
render.resolution_y = resolution
render.resolution_percentage = 100
bpy.context.scene.cycles.filter_width = 0.01
bpy.context.scene.render.film_transparent = True

bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.diffuse_bounces = 1
bpy.context.scene.cycles.glossy_bounces = 1
bpy.context.scene.cycles.transparent_max_bounces = 3
bpy.context.scene.cycles.transmission_bounces = 3
bpy.context.scene.cycles.samples = 32
bpy.context.scene.cycles.use_denoising = True

# CUDA 장치 활성화 함수
def enable_cuda_devices():
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()

    # GPU 장치 유형 설정 시도
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print("Compute device selected: {0}".format(compute_device_type))
            break
        except TypeError:
            pass

    # CUDA/OPENCL 장치가 있는지 확인
    acceleratedTypes = ['CUDA', 'OPENCL']
    accelerated = any(device.type in acceleratedTypes for device in cprefs.devices)
    print('Accelerated render = {0}'.format(accelerated))

    # CUDA/OPENCL 장치가 있으면 해당 장치만 활성화, 없으면 모든 장치 활성화
    print(cprefs.devices)
    for device in cprefs.devices:
        device.use = not accelerated or device.type in acceleratedTypes
        print('Device enabled ({type}) = {enabled}'.format(type=device.type, enabled=device.use))

    return accelerated

enable_cuda_devices()
context.active_object.select_set(True)
bpy.ops.object.delete()

# 텍스처가 있는 메쉬 가져오기
bpy.ops.object.select_all(action='DESELECT')

# 객체의 경계를 가져오는 함수
def bounds(obj, local=False):
    local_coords = obj.bound_box[:]
    om = obj.matrix_world

    if not local:
        worldify = lambda p: om @ Vector(p[:])
        coords = [worldify(p).to_tuple() for p in local_coords]
    else:
        coords = [p[:] for p in local_coords]

    rotated = zip(*coords[::-1])

    push_axis = []
    for (axis, _list) in zip('xyz', rotated):
        info = lambda: None
        info.max = max(_list)
        info.min = min(_list)
        info.distance = info.max - info.min
        push_axis.append(info)

    import collections

    originals = dict(zip(['x', 'y', 'z'], push_axis))

    o_details = collections.namedtuple('object_details', 'x y z')
    return o_details(**originals)

# Blender에서 카메라 변환 행렬을 가져오는 함수
def get_3x4_RT_matrix_from_blender(cam):
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1*R_world2bcam @ location

    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

# OBJ 파일 가져오기
bpy.ops.import_scene.obj(filepath=obj_path, use_edges=False, use_smooth_groups=False, split_mode='OFF')

for this_obj in bpy.data.objects:
    if this_obj.type == "MESH":
        this_obj.select_set(True)
        bpy.context.view_layer.objects.active = this_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.split_normals()

bpy.ops.object.mode_set(mode='OBJECT')
obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

mesh_obj = obj
factor = max(mesh_obj.dimensions[0], mesh_obj.dimensions[1], mesh_obj.dimensions[2]) / scale
mesh_obj.scale[0] /= factor
mesh_obj.scale[1] /= factor
mesh_obj.scale[2] /= factor
bpy.ops.object.transform_apply(scale=True)

# 객체 중심 정렬
obj_bounds = bounds(mesh_obj)
obj_center = Vector((
    (obj_bounds.x.max + obj_bounds.x.min) / 2,
    (obj_bounds.y.max + obj_bounds.y.min) / 2,
    (obj_bounds.z.max + obj_bounds.z.min) / 2,
))
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
mesh_obj.location -= obj_center

factor = max(mesh_obj.dimensions[0], mesh_obj.dimensions[1], mesh_obj.dimensions[2]) / scale
mesh_obj.scale[0] /= factor
mesh_obj.scale[1] /= factor
mesh_obj.scale[2] /= factor
bpy.ops.object.transform_apply(scale=True)

# 조명 추가
bpy.ops.object.light_add(type='AREA')
light2 = bpy.data.lights['Area']

light2.energy = 30000
bpy.data.objects['Area'].location = (0, 0, 1.2)  # 조명을 모델 앞쪽에 배치

bpy.data.objects['Area'].scale[0] = 100
bpy.data.objects['Area'].scale[1] = 100
bpy.data.objects['Area'].scale[2] = 100

# 카메라 배치
cam = scene.objects['Camera']
cam.location = (0, 3, 0)  # 초기 위치 설정
cam.data.lens = 100
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

stepsize_horizontal = 90.0 / (views - 1)  # -45도에서 45도까지의 각도를 나누기 위해
# stepsize_vertical = 90.0 / (views - 1)  # -45도에서 45도까지의 각도를 나누기 위해
stepsize_vertical = 60.0 / (views - 1)  # -45도에서 45도까지의 각도를 나누기 위해

model_identifier = os.path.split(os.path.split(obj_path)[0])[1]
synset_idx = obj_path.split('/')[-3]

img_folder = os.path.join(os.path.abspath(output_folder), 'img')

os.makedirs(img_folder, exist_ok=True)

# transform.json 생성
to_export = {
    'camera_angle_x': bpy.data.cameras[0].angle_x,
    "aabb": [[-scale/2,-scale/2,-scale/2],
            [scale/2,scale/2,scale/2]]
}
frames = [] 

for k in range(views):
    for i in range(views):
        angle_horizontal = -45 + stepsize_horizontal * i + 180
        # angle_vertical = -45 + stepsize_vertical * k
        angle_vertical = -30 + stepsize_vertical * k
        radians_horizontal = math.radians(angle_horizontal)
        radians_vertical = math.radians(angle_vertical)
        cam_empty.rotation_euler[2] = radians_horizontal
        cam_empty.rotation_euler[0] = radians_vertical

        render_file_path = os.path.join(img_folder, '%03d.png' % (views*k+i))
        scene.render.filepath = render_file_path
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        rt = get_3x4_RT_matrix_from_blender(cam)
        pos, rt, scale = cam.matrix_world.decompose()
        rt = rt.to_matrix()

        matrix = []
        for ii in range(3):
            a = []
            for jj in range(3):
                a.append(rt[ii][jj])
            a.append(pos[ii])
            matrix.append(a)
        matrix.append([0, 0, 0, 1])

        to_add = {
            "file_path": f'{str(i).zfill(3)}.png',
            "transform_matrix": matrix
        }
        frames.append(to_add)

to_export['frames'] = frames
with open(f'{img_folder}/transforms.json', 'w') as f:
    json.dump(to_export, f, indent=4)
