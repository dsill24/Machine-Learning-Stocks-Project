import torch.nn
from LSTM import fromScratchLSTM

class Nueral_Network(torch.nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int, output_dim: int = 1):
        super().__init__()
        self.lstm = fromScratchLSTM(input_size, hidden_layer_size)
        self.fc = torch.nn.Linear(hidden_layer_size, output_dim)

    def forward(self, current_input):
        new_output, (hidden_gate_n_at_time_t, new_state_n_at_time_t) = self.lstm(current_input)
        output = self.fc(new_output[:,-1,:]) # save only the last time step
        return output
    