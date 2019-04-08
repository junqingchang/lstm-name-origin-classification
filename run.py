from utils import *
from model import *
from train import *
from test import *
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

################################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
n_hidden = 64
n_layers = 1
learning_rate = 0.01
n_iters = 100000
print_every = 5000
plot_every = 1000
batch_size = 1
test_size = 1000
################################


def main():
    all_categories, category_lines = getAllCategories('data/names/*.txt')
    n_categories = len(all_categories)

    rnn = LSTMClassifier(n_hidden, getNumLetters(), n_categories, n_layers)
    rnn.to(device)

    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.1)

    current_loss = 0
    all_losses = []
    test_losses = []

    start = time.time()
    for iter in range(1, n_iters + 1):

        loss = train(all_categories, category_lines, batch_size, rnn, optimizer, device)
        current_loss += loss
        

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            curr_test_loss = 0
            for i in range(test_size):
                loss, line, guess, correct = val(all_categories, category_lines, batch_size, rnn, device)
                curr_test_loss += loss
            curr_test_loss /= test_size
            test_losses.append(curr_test_loss)
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.title('Train Loss')
    plt.ylabel('loss')
    plt.plot(all_losses)
    plt.show()

    plt.figure()
    plt.title('Test Loss')
    plt.ylabel('loss')
    plt.plot(test_losses)
    plt.show()

    # If you wish to save the model
    # torch.save(rnn, 'lstm-classifier.pt')

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000
    total_correct = 0

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines, batch_size)
        line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)
        output = evaluate(line_tensor, rnn)
        guesses = categoryFromOutput(output, all_categories)
        for i in range(len(guesses)):
            guess, guess_i = guesses[i]
            category_i = all_categories.index(category[i])
            confusion[category_i][guess_i] += 1
            if(guesses[i][0]==category[i]):
                total_correct += 1
    
    print('Accuracy of Model: {}/{} ({}%)'.format(total_correct, n_confusion*batch_size, 100*total_correct/(n_confusion*batch_size)))

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

if __name__ == "__main__":
    main()