import torch
import torch.nn.functional as F
from utils import *

def train(all_categories, category_lines, batch_size, rnn, optimizer, device):
    rnn.train()
    optimizer.zero_grad()

    category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines, batch_size)
    line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)
    output = rnn(line_tensor)
    loss = F.nll_loss(output, category_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()