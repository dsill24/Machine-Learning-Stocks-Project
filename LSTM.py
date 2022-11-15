import math
import torch
import torch.nn


class fromScratchLSTM(torch.nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int):
        super().__init__() ## inputs from torch nn module the base constructor

        #input gate
        self.input_gate_input_weights = torch.nn.parameter(torch.tensor(input_size, hidden_layer_size))
        self.input_gate_hidden_weights = torch.nn.parameter(torch.tensor(input_size, hidden_layer_size))
        self.input_gate_bias = torch.nn.parameter(torch.tensor(hidden_layer_size))
        
        #forget gate
        self.forget_gate_input_weights = torch.nn.parameter(torch.tensor(input_size, hidden_layer_size))
        self.forget_gate_hidden_weights = torch.nn.parameter(torch.tensor(input_size, hidden_layer_size))
        self.forget_gate_bias = torch.nn.parameter(torch.tensor(hidden_layer_size))

        #cell gate
        self.cell_gate_input_weights = torch.nn.parameter(torch.tensor(input_size, hidden_layer_size))
        self.cell_gate_hidden_weights = torch.nn.parameter(torch.tensor(input_size, hidden_layer_size))
        self.cell_gate_bias = torch.nn.parameter(torch.tensor(hidden_layer_size))

        #output gate
        self.output_gate_input_weights = torch.nn.parameter(torch.tensor(input_size, hidden_layer_size))
        self.output_gate_hidden_weights = torch.nn.parameter(torch.tensor(input_size, hidden_layer_size))
        self.output_gate_bias = torch.nn.parameter(torch.tensor(hidden_layer_size))

        self.init_weights() #intialize the weights of the cell

    # fill every weight in the LSTM parameters with a random number sampled from a continous uniform distribution
    # from the equation 1 / stdv + stdv
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_layer_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def feedforward(self, x, init_states = None):
        """
        x.shape gives (batch_size, sequence_size, input_size)
        """

        batch_size, sequence_size, _ = x.size()
        hidden_sequence = []

        if init_states is None: 
            hidden_gate_at_time_t, new_state_at_time_t = (
                torch.zeros(batch_size, self.hidden_layer_size).to(x.device),
                torch.zeros(batch_size, self.hidden_layer_size).to(x.device),
            )
        else:
            hidden_gate_at_time_t, new_state_at_time_t = init_states

        for time in range(sequence_size):
            current_input_at_time_t = x[:, time, :]

            input_gate_at_time_t = torch.sigmoid(current_input_at_time_t @ self.input_gate_input_weights
            + hidden_gate_at_time_t @ self.input_gate_hidden_weights + self.input_gate_bias)

            forget_gate_at_time_t = torch.sigmoid(current_input_at_time_t @ self.forget_gate_input_weights
            + hidden_gate_at_time_t @ self.forget_gate_hidden_weights + self.forget_gate_bias)

            cell_gate_at_time_t = torch.tanh(current_input_at_time_t @ self.cell_gate_input_weights
            + hidden_gate_at_time_t @ self.cell_gate_hidden_weights + self.cell_gate_bias)

            output_gate_at_time_t = torch.sigmoid(current_input_at_time_t @ self.output_gate_input_weights
            + hidden_gate_at_time_t @ self.output_gate_hidden_weights + self.output_gate_bias)

            new_state_at_time_t = forget_gate_at_time_t * new_state_at_time_t + input_gate_at_time_t * cell_gate_at_time_t

            hidden_gate_at_time_t = output_gate_at_time_t * torch.tanh(new_state_at_time_t)

        hidden_sequence = torch.cat(hidden_sequence, dim=0)
        hidden_sequence = hidden_sequence.transpose(0, 1).contiguous()
        return hidden_sequence, (hidden_gate_at_time_t, new_state_at_time_t)

