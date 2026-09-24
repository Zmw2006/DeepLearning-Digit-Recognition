import matplotlib.pyplot as plt


def draw_curve(loss_list, acc_list):
    plt.figure()
    plt.plot(loss_list)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/loss.png')

    plt.figure()
    plt.plot(acc_list)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('results/accuracy.png')
