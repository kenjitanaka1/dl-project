import os
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    model_folder = 'saved_models'
    i = 0
    accs = []
    losses = []
    while os.path.exists(os.path.join(model_folder, f'loss_epoch_{i}.npy')):
        acc, loss = np.load(os.path.join(model_folder, f'loss_epoch_{i}.npy'))
        accs.append(acc)
        losses.append(loss)
        # print(acc)
        i += 1

    print(accs)
    print(losses)
    plt.plot(accs)
    plt.title('Accuracy vs Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((0,1))
    plt.xlabel('Epoch')
    plt.savefig('accuracies.png')



    plt.figure()
    plt.plot(losses)
    plt.title('Loss vs Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('losses.png')
    # plt.show()