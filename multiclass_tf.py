import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def classify_text(model, vectorizer, input_text):
    # テキストの前処理
    text_data = vectorizer.transform([input_text]).toarray()
    text_data = preprocessing.sequence.pad_sequences(text_data, maxlen=max_length)

    # 分類
    predictions = model.predict(text_data)
    predicted_label = np.argmax(predictions, axis=-1)

    # ジャンル名を取得
    class_name = newsgroups.target_names[predicted_label[0]]
    return class_name


# ハイパーパラメータ
vocab_size = 10000
max_length = 256
embedding_dim = 128
num_classes = 20
num_epochs = 10
batch_size = 64

# データセットのロード
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts, labels = newsgroups.data, newsgroups.target

# 訓練データとテストデータに分割
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# テキスト処理
vectorizer = CountVectorizer(max_features=vocab_size)
train_data = vectorizer.fit_transform(train_texts).toarray()
test_data = vectorizer.transform(test_texts).toarray()

# パディング
train_data = preprocessing.sequence.pad_sequences(train_data, maxlen=max_length)
test_data = preprocessing.sequence.pad_sequences(test_data, maxlen=max_length)

# モデル構築
model = models.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# コンパイル
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# コールバック定義
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# 訓練
history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

# 評価
test_loss, test_acc = model.evaluate(test_data, test_labels)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


#input_text = "The team played well and won the match. It was a great game."
#output_class = classify_text(model, vectorizer, input_text)
#print(f"Input Text: {input_text}")
#print(f"Output Class: {output_class}")
