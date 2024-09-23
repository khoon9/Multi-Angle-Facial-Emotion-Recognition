import bpy
import argparse
from mathutils import Vector, Matrix
import numpy as np
import json
import os
import math

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

def obj_to_png(obj_path,file_num,output_folder):

    # 설정 값 직접 정의
    views_horizontal = 13
    views_vertical = 9

    scale = 1  # 모델에 적용될 스케일링 인수
    file_format = 'PNG'  # 생성될 파일의 형식
    resolution = 224  # 이미지 해상도
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


    enable_cuda_devices()

    # 기존 조명 삭제
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    context.active_object.select_set(True)
    bpy.ops.object.delete()

    # 텍스처가 있는 메쉬 가져오기
    bpy.ops.object.select_all(action='DESELECT')

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


    # # POINT 조명 추가
    # bpy.ops.object.light_add(type='POINT')
    # light_point = bpy.data.lights['Point']
    # light_point.energy = 1000
    # bpy.data.objects['Point'].location = (0, 0, 2)

    # # SUN 조명 추가. 위에서 내리쬐는 햇빛과 같은 조명 기능 수행
    # bpy.ops.object.light_add(type='SUN')
    # light_sun = bpy.data.lights['Sun']
    # light_sun.energy = 2
    # bpy.data.objects['Sun'].location = (0, 0, 10)



    # 조명 추가
    bpy.ops.object.light_add(type='AREA')
    light2 = bpy.data.lights['Area']

    # light2.energy = 30000
    light2.energy = 30000
    bpy.data.objects['Area'].location = (0, 0, 1.2)  # 조명을 모델 앞쪽에 배치

    bpy.data.objects['Area'].scale[0] = 100
    bpy.data.objects['Area'].scale[1] = 100
    bpy.data.objects['Area'].scale[2] = 100

    # 카메라 배치
    cam = scene.objects['Camera']
    cam.location = (0, 4, 0)  # 초기 위치 설정
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

    stepsize_horizontal = 120.0 / (views_horizontal - 1)  # -60도에서 60도까지의 각도를 나누기 위해
    stepsize_vertical = 80.0 / (views_vertical - 1)  # -40도에서 40도까지의 각도를 나누기 위해

    img_folder = os.path.join(os.path.abspath(output_folder), 'img')

    os.makedirs(img_folder, exist_ok=True)

    for i in range(views_vertical):
        for k in range(views_horizontal):
            angle_horizontal = -60 + stepsize_horizontal * k + 180
            # angle_horizontal = -30 + stepsize_horizontal * k + 180
            angle_vertical = -40 + stepsize_vertical * i
            # angle_vertical = -30 + stepsize_vertical * i
            radians_horizontal = math.radians(angle_horizontal)
            radians_vertical = math.radians(angle_vertical)
            cam_empty.rotation_euler[2] = radians_horizontal
            cam_empty.rotation_euler[0] = radians_vertical

            render_file_path = os.path.join(img_folder, '%06d.png' % (file_num))
            scene.render.filepath = render_file_path
            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()
            
            file_num += 1

# 메인 함수
def main():
    import sys
    argv = sys.argv

    if "--" not in argv:
        argv = []  # 정상적인 실행이 아닌 경우, 빈 리스트로 초기화
    else:
        argv = argv[argv.index("--") + 1:]  # "--" 이후의 인수만 가져옴

    parser = argparse.ArgumentParser(description="Blender OBJ to PNG")
    parser.add_argument("file_path", type=str, help="Path to the OBJ file")
    parser.add_argument("file_num", type=int, help="Starting file number")
    parser.add_argument("output_dir", type=str, help="output path")
    args = parser.parse_args(argv)

    obj_to_png(args.file_path, args.file_num, args.output_dir)

if __name__ == "__main__":
    main()

