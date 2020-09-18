import torch
import torch.nn as nn
from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, line_to_tensor

class RNN(nn.Module):
    def __init__(self, input_sz, hidden_sz, output_sz):
        super(RNN, self).__init__()

        self.hidden_sz = hidden_sz
        self.i2h = nn.Linear(input_sz + hidden_sz, hidden_sz)
        self.i2o = nn.Linear(input_sz + hidden_sz, output_sz)
        self.softmax = nn.LogSoftmax(dim=1) 


    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_sz)


category_lines, all_categories = load_data()
n_categories = len(all_categories)

PATH = "rnn_ffn.pth"
n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)
rnn.load_state_dict(torch.load(PATH))
rnn.eval()


def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

def get_predication(name):
    print(f"\n> {name} ")

    with torch.no_grad():
        line_tensor = line_to_tensor(name)

        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output)

        print(guess)
        return guess
        