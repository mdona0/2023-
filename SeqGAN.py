import torch
import torch.nn as nn
import torch.optim as optim

# ハイパーパラメータ
vocab_size = 5000
embed_size = 128
hidden_size = 256
seq_length = 20
batch_size = 64
num_epochs = 50
generator_lr = 0.001
discriminator_lr = 0.001

# ジェネレータ
class Generator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_length):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        x, h = self.lstm(x, h)
        x = self.linear(x)
        return x, h

# ディスクリミネータ
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_length):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h=None):
        x = self.embed(x)
        x, h = self.lstm(x, h)
        x = self.linear(x[:, -1])
        return x

# 損失関数と最適化アルゴリズム
generator = Generator(vocab_size, embed_size, hidden_size, seq_length)
discriminator = Discriminator(vocab_size, embed_size, hidden_size, seq_length)
g_optimizer = optim.Adam(generator.parameters(), lr=generator_lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=discriminator_lr)
criterion = nn.BCEWithLogitsLoss()

# 訓練ループ
for epoch in range(num_epochs):
    # ここではダミーのデータを使用しています。実際には、適切なデータセットをロードして使用してください。
    real_data = torch.randint(0, vocab_size, (batch_size, seq_length)).long()
    noise = torch.randint(0, vocab_size, (batch_size, seq_length)).long()

    # ジェネレータの訓練
    g_optimizer.zero_grad()
    fake_data, _ = generator(noise)
    fake_logits = discriminator(fake_data)
    g_loss = criterion(fake_logits, torch.ones_like(fake_logits))
    g_loss.backward()
    g_optimizer.step()

 # ディスクリミネータの訓練
    d_optimizer.zero_grad()
    real_logits = discriminator(real_data)
    fake_logits = discriminator(fake_data.detach())
    
    real_loss = criterion(real_logits, torch.ones_like(real_logits))
    fake_loss = criterion(fake_logits, torch.zeros_like(fake_logits))
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    d_optimizer.step()

    # エポックごとの損失を表示
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Generator Loss: {g_loss.item():.4f}, "
              f"Discriminator Loss: {d_loss.item():.4f}")
