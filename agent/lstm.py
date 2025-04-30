import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = 64
        self.num_layers = 1
        self.output_size = 3
        self.lstm = nn.LSTM(input_size, self.hidden_size ,self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def init_hidden(self,batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h0,c0
    
    def forward(self, x, h, c):
        lstm_out, h, c = self.lstm(x, h, c)
        output = self.linear(lstm_out[:, -1, :])
        return output, h, c
    
# input_size = 1
# hidden_size = 300
# num_layers = 1
# output_size = 1

# model = LSTM(input_size, hidden_size, num_layers, output_size)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 300
# burn_steps = 50
# losses = []
# for epoch in range(num_epochs):
#     for seq, labels in dataloader:
#         seq = seq.unsqueeze(-1)  # (batch_size, seq_length, input_size)
#         labels = labels.unsqueeze(-1)  # (batch_size, output_size)
        
#         hc = model.init_hidden(seq.size(0))
        
#         with torch.no_grad():
#             for _ in range(burn_steps):
#              _, hc = model(seq, hc)
             
#         outputs,hc = model(seq,hc)
#         loss = criterion(outputs, labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     losses.append(loss.item())
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# current_time = datetime.datetime.now().strftime('%Y-%m%d_%H%M')

# loss_plot_filename = f'plot/plot_loss_{current_time}.png'
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.yscale('log')
# plt.savefig(loss_plot_filename)
# plt.show()

# outputs = []
# truth = []
# for seq, labels in dataset:
#     seq = seq.unsqueeze(-1)  # (batch_size, length, input_size)
#     truth.append( labels.detach().numpy() ) 
#     hc = model.init_hidden(1)  # バッチサイズ1の隠れ状態を初期化
#     with torch.no_grad():
#         output, hc = model(seq.unsqueeze(0), hc)
#     outputs.append(output.squeeze().detach().numpy() )

# outputs = np.array( outputs )
# truth   = np.array( truth )
# plot_length = min(len(outputs), len(truth), 24*10)

# prediction_plot_filename = f'plot/plot_predictions_{current_time}.png'
# plt.plot(range(plot_length), outputs[:plot_length], label='NN')
# plt.plot(range(plot_length), truth[:plot_length], label='true')
# plt.legend()
# plt.savefig(prediction_plot_filename) 
# plt.show()
    
# model.eval()
# with torch.no_grad():
#     test_inputs = data[length:]
#     test_inputs = torch.tensor(test_inputs, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
#     hc = model.init_hidden(1)
#     test_output, _ = model(test_inputs,hc)
#     print(f'Predicted Sunshine: {test_output.item()}')



