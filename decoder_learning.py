# model 학습
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_embeddings_and_labels(emb_file='data/npy_files/X_embeddings.npy', label_file='data/npy_files/y_labels.npy'):
    X_embeddings = np.load(emb_file)
    y_labels = np.load(label_file)
    return X_embeddings, y_labels

def decoder_learning_function():
    # 데이터 준비
    X, y = load_embeddings_and_labels()  

    # 원-핫 인코딩
    y_onehot = to_categorical(y, num_classes=4)

    # 학습 및 검증 데이터셋으로 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.3, random_state=1, stratify=y)

    model = Sequential()
    model.add(Dense(256, input_dim=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 체크포인트 콜백 설정
    checkpoint_filepath = 'models/model_checkpoint.h5'
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_accuracy', mode='max')

    # 조기 종료 콜백 설정
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    history = model.fit(X_train, y_train, 
                        epochs=100, 
                        batch_size=32, 
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint_callback])

    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f'Validation Accuracy: {100*accuracy:.2f}%')

    return model, history

# def main():
#     decoder_learning_function()

# if __name__ == "__main__":
#     main()

mlp_model, mlp_history = decoder_learning_function()
# 정확도 시각화
plt.plot(mlp_history.history['accuracy'])
plt.plot(mlp_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 손실 시각화
plt.plot(mlp_history.history['loss'])
plt.plot(mlp_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()