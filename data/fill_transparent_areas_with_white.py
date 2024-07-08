from PIL import Image
import os

def convert_transparent_to_white(img_path):
    img = Image.open(img_path).convert("RGBA")
    white_background = Image.new("RGBA", img.size, "WHITE")
    white_background.paste(img, (0, 0), img)
    white_background = white_background.convert("RGB")
    white_background.save(img_path)

img_folder = './output/img'  # Blender 스크립트에서 설정한 경로와 일치하도록 변경
for img_name in os.listdir(img_folder):
    if img_name.endswith(".png"):
        convert_transparent_to_white(os.path.join(img_folder, img_name))
