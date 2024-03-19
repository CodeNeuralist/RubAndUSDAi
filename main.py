import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

data = yf.download('USDRUB=X', start='2023-03-18', end='2024-03-18')

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

scaled_data = torch.FloatTensor(scaled_data).view(-1)

input_size = 1
hidden_size = 64
output_size = 1
num_layers = 2
seq_length = 7

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x.view(x.size(0), -1, 1), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN(input_size, hidden_size, output_size, num_layers).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    for i in range(len(scaled_data) - seq_length):
        seq = scaled_data[i:i+seq_length].view(1, -1)
        seq = seq.unsqueeze(2).to(device)
        label = scaled_data[i+seq_length:i+seq_length+1].to(device)

        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_seq = scaled_data[-seq_length:].view(1, -1)
    test_seq = test_seq.unsqueeze(2).to(device)
    pred = model(test_seq)
    pred = scaler.inverse_transform(pred.cpu().numpy())

print("Predicted value for tomorrow's USD to RUB exchange rate:", pred.item())
