# demo
import cv2
import torch
from torchvision import transforms
from edgeface.face_alignment import align
from edgeface.backbones import get_model
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def load_model_function(checkpoint_filepath):
    # 모델 체크포인트 불러오기
    loaded_model = load_model(checkpoint_filepath)

    return loaded_model

def emb_model_load():
    # 모델 로드
    model_name = "edgeface_s_gamma_05"  # 또는 edgeface_xs_gamma_06
    model = get_model(model_name)
    checkpoint_path = f'edgeface/checkpoints/{model_name}.pt'
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model

def png_to_emb_vector_one(path,model,img):
    # 이미지 변환 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 이미지 로드 및 정렬
    aligned = align.get_aligned_face(path,img)  # 정렬된 얼굴

    if aligned==None:
        return None

    # 이미지 변환 및 배치 차원 추가
    transformed_input = transform(aligned).unsqueeze(0)  # 변환 및 배치 차원 추가

    # aligned_img = Image.fromarray(np.uint8(aligned)).convert('RGB')
    aligned_np = np.array(aligned)
    # 색상 채널을 RGB에서 BGR로 변환
    aligned_np = cv2.cvtColor(aligned_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("face detection",aligned_np)

    # 임베딩 추출
    with torch.no_grad():
        embedding = model(transformed_input)

    return embedding

def cam_to_prediction_function():
    # 레이블 매핑 정의
    labels_map = {
        0: "Neutral",
        1: "Happy",
        2: "Angry",
        3: "Surprised"
    }
    # 모델 불러오기
    emb_model = emb_model_load()
    model = load_model_function('models/model_checkpoint_60_40degree_100epochs_black_and_black_sun.h5')
    # 웹 캠 연결 및 마스크 적용하기
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()
    num = 0
    try:
        while webcam.isOpened():
            status, frame = webcam.read()
            if status:
                #마스크와 이미지 크기 맞춰 주기
                # frame=cv2.resize(frame, (960, 720))
                # cv2.imwrite('cam/temp/recent.png',frame)
                # OpenCV의 BGR 형식을 RGB 형식으로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # NumPy 배열을 PIL 이미지로 변환
                img = Image.fromarray(frame_rgb)

                # 얼굴 인식 및 예외처리
                sample = png_to_emb_vector_one('',emb_model,img)
                if sample==None:
                    continue
                
                num += 1

                # 예측된 클래스 및 레이블 변환
                prediction = model.predict(sample.numpy())
                predicted_class = np.argmax(prediction, axis=1)[0]
                predicted_label = labels_map[predicted_class]

                # 프레임에 예측 결과 추가
                cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 프레임 표시
                cv2.imshow("frame", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()

cam_to_prediction_function()