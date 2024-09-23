from PIL import Image
import os

def convert_transparent_to_color(img_folder, img_name, output_folder, color):
    img_path = os.path.join(img_folder, img_name)
    img = Image.open(img_path).convert("RGBA")
    color_background = Image.new("RGBA", img.size, color)
    color_background.paste(img, (0, 0), img)
    color_background = color_background.convert("RGB")
    color_background.save(os.path.join(output_folder, img_name))

img_folder = './output/img_60_40degree_black_sun'  # Blender 스크립트에서 설정한 경로와 일치하도록 변경
for img_name in os.listdir(img_folder):
    if img_name.endswith(".png"):
        convert_transparent_to_color(img_folder, img_name, './output/img',"BLACK")
