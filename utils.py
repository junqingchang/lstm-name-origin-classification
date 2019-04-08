from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import random
import time
import math
from torch.nn.utils.rnn import pack_sequence
from copy import deepcopy


def findFiles(path): return glob.glob(path)

def getAllLetters(): return string.ascii_letters + " .,;'"

def getNumLetters(): return len(getAllLetters())

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in getAllLetters()
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def getAllCategories(path):
    category_lines = {}
    all_categories = []
    for filename in findFiles(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    return all_categories, category_lines

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return getAllLetters().find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, getNumLetters())
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(lines, batch_size):
    line_values = []
    for line in lines:
        tensor = torch.zeros(len(line), getNumLetters())
        for li, letter in enumerate(line):
            tensor[li][letterToIndex(letter)] = 1
        line_values.append(deepcopy(tensor))
    output_tensor = pack_sequence(line_values)
    return output_tensor

def categoryFromOutput(output, all_categories):
    categories = []
    for each_output in output:
        top_n, top_i = each_output.topk(1)
        category_i = top_i[0].item()
        categories.append((all_categories[category_i], category_i))
    return categories

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(all_categories, category_lines, batch_size):
    o_category = []
    lines = []
    categories = []
    for i in range(batch_size):
        category = randomChoice(all_categories) 
        o_category.append(category)
        line = randomChoice(category_lines[category])
        lines.append(line)
        categories.append(all_categories.index(category))

    lines_copy = deepcopy(lines)
    lines.sort(key=len)
    lines.reverse()
    lines_idx = [lines_copy.index(x) for x in lines]
    sorted_category = []
    sorted_o_category = []
    for idx in lines_idx:
        sorted_category.append(categories[idx])
        sorted_o_category.append(o_category[idx])

    line_tensor = lineToTensor(lines, batch_size)
    category_tensor = torch.tensor(sorted_category)
    return sorted_o_category, lines, category_tensor, line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
