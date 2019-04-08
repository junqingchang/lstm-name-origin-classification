import torch
import torch.nn.functional as F
from utils import *

def evaluate(line_tensor, rnn):
    rnn.eval()
    with torch.no_grad():
        output = rnn(line_tensor)

    return output

def val(all_categories, category_lines, batch_size, rnn, device):
    rnn.eval()
    with torch.no_grad():
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines, batch_size)
        line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)

        output = rnn(line_tensor)

        loss = F.nll_loss(output, category_tensor)

        guess = categoryFromOutput(output, all_categories)
        for i in range(len(guess)):
            if(guess[i][0]==category[i]):
                correct = '✓'
            else:
                correct = '✗ (%s)' % category
                break
    return loss.item(), line, guess, correct