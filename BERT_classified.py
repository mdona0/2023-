import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 事前学習済みのBERTモデルとトークナイザをロード
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 分類する文章
sentences = [
    "I had an amazing time at the park today.",
    "The movie was a waste of time, I didn't like it at all."
]

# 文章をトークン化し、テンソルに変換
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# モデルの評価モードを設定
model.eval()

# GPUが利用可能な場合はGPUを使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs.to(device)

# モデルの出力を取得
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 結果を確率に変換
probs = torch.softmax(logits, dim=-1)

# ラベルが0の場合はネガティブ、1の場合はポジティブ
labels = ["Negative", "Positive"]

# 結果を表示
for i, sentence in enumerate(sentences):
    label_index = probs[i].argmax().item()
    confidence = probs[i][label_index].item()
    print(f"Sentence: {sentence}\nLabel: {labels[label_index]} (confidence: {confidence:.2f})\n")
