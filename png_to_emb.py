# png 임베딩 
import os
import torch
from torchvision import transforms
from edgeface.face_alignment import align
from edgeface.backbones import get_model
from PIL import Image
import numpy as np

def load_and_align_images(image_paths):
    aligned_images = []
    y = []
    for idx, path in enumerate(image_paths):
        aligned = align.get_aligned_face(path)  # 정렬된 얼굴
        if aligned is None:
            print(f"Warning: Could not align image at path {path}")
        else:
            aligned_images.append(aligned)
            y.append((idx)//55809)
    return aligned_images, y

def transform_images(images, transform):
    transformed_images = []
    for img in images:
        if isinstance(img, Image.Image):
            img_tensor = transform(img).unsqueeze(0)
            transformed_images.append(img_tensor)
        else:
            print("Error: Image is not a PIL.Image instance")
    return torch.cat(transformed_images)

def emb_model_load():
    # 모델 로드
    model_name = "edgeface_s_gamma_05"  # 또는 edgeface_xs_gamma_06
    model = get_model(model_name)
    checkpoint_path = f'edgeface/checkpoints/{model_name}.pt'
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model

def png_to_emb_vector(image_paths,model):
    # 이미지 변환 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 이미지 로드 및 정렬
    aligned_images, y = load_and_align_images(image_paths)
    
    if not aligned_images:
        print("Error: No images were aligned successfully.")
        return

    # 이미지 변환 및 배치 차원 추가
    transformed_inputs = transform_images(aligned_images, transform)

    # 임베딩 추출
    with torch.no_grad():
        embeddings = model(transformed_inputs)

    return embeddings, y

def png_to_emb_function():
    # 모델 불러오기
    emb_model = emb_model_load()

    image_dir = 'data/output/img'
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('png'))]
    image_files
    # 이미지 로드 및 출력 예제
    images = []
    for image_file in image_files:
        img = os.path.join(image_dir, image_file)
        images.append(img)
    image_paths = images
    X_tensor, y = png_to_emb_vector(image_paths,emb_model)
    return X_tensor, y

X_tensor, y = png_to_emb_function()
X = X_tensor.numpy()
y = np.array(y)

def save_embeddings_and_labels(embeddings, labels, emb_file='data/mlp_input/npy_files/X_embeddings.npy', label_file='data/mlp_input/npy_files/y_labels.npy'):
    np.save(emb_file, embeddings)
    np.save(label_file, labels)

save_embeddings_and_labels(X, y)