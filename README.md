# Multi-Angle-Facial-Emotion-Recognition

## 설정

### 1. 가상 환경 생성

가상 환경을 생성하기 위해 아래 명령어를 실행합니다.

```sh
python3.8 -m venv ./env
```

### 2. 가상 환경 실행

가상 환경을 활성화하기 위해 아래 명령어를 실행합니다.

```sh
source env/bin/activate
```

### 3. 사전정의 패키지 구성 적용

필요한 패키지를 설치하기 위해 아래 명령어를 실행합니다.

```sh
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html timm mxnet opencv-python tensorflow scikit-learn matplotlib
```

## 4. 실행 가이드

```sh
python emotion_recognition_cam.py
```

- 실행 전 IDE 재부팅 권장
