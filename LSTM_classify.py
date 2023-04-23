import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data, datasets

# ハイパーパラメータ
vocab_size = 10000
embed_size = 128
hidden_size = 256
num_classes = 2
num_epochs = 10
batch_size = 64
lr = 0.001

# 文章とラベルのフィールドを定義
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True)
LABEL = data.LabelField(dtype=torch.float)

# IMDBデータセットをロード
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split()

# 語彙を構築
TEXT.build_vocab(train_data, max_size=vocab_size)
LABEL.build_vocab(train_data)

# データローダを作成
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True
)

# LSTMモデル
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

# モデルと最適化アルゴリズムを初期化
model = LSTMClassifier(vocab_size, embed_size, hidden_size, num_classes)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# GPUが利用可能な場合はGPUを使う
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

# 訓練ループ
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    for batch in train_iterator:
        text, label = batch.text.to(device), batch.label.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()

        _, preds = torch.max(output, 1)
        acc = (preds == label.long()).sum() / len(label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_iterator):.4f}, Acc: {epoch_acc / len(train_iterator):.4f}")