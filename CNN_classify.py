import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import spacy

# ハイパーパラメータ
vocab_size = 10000
embed_size = 128
num_filters = 100
filter_sizes = [3, 4, 5]
num_classes = 2
num_epochs = 10
batch_size = 64
lr = 0.001

# データセット
train_data, test_data = IMDB(split=('train', 'test'))
train_data, valid_data = torch.utils.data.random_split(train_data, [20000, 5000])

# テキスト処理
tokenizer = get_tokenizer("spacy", "en_core_web_sm")
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 語彙
vocab = build_vocab_from_iterator(yield_tokens(train_data), max_size=vocab_size)

# データローダ
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)

def collate_batch(batch):
    text_list, label_list = [], []
    for (_label, _text) in batch:
        text_list.append(torch.tensor(text_pipeline(_text), dtype=torch.int64))
        label_list.append(torch.tensor(label_pipeline(_label), dtype=torch.int64))
    return torch.nn.utils.rnn.pad_sequence(text_list, padding_value=0, batch_first=True), torch.tensor(label_list)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# CNNモデル
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_size)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embed_size)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, seq_len - filter_size + 
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, num_filters), ...]*len(filter_sizes)
        x = torch.cat(x, 1)  # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)  # (batch_size, num_classes)
        return x

# モデルと最適化アルゴリズムを初期化
model = TextCNN(vocab_size, embed_size, num_filters, filter_sizes, num_classes)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 訓練ループ
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for text, label in train_dataloader:
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()

        _, preds = torch.max(output, 1)
        acc = (preds == label.long()).sum() / len(label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}, Acc: {epoch_acc / len(train_dataloader):.4f}")

# テストデータでの評価
test_loss = 0
test_acc = 0
model.eval()
with torch.no_grad():
    for text, label in test_dataloader:
        output = model(text)
        loss = criterion(output, label.long())

        _, preds = torch.max(output, 1)
        acc = (preds == label.long()).sum() / len(label)
        test_loss += loss.item()
        test_acc += acc.item()

print(f"Test Loss: {test_loss / len(test_dataloader):.4f}, Test Acc: {test_acc / len(test_dataloader):.4f}")
