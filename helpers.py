import matplotlib.pyplot as plt
import numpy as np


def loss_graph(train_loss, test_loss, accuracy,iteration, optim_name='Geoadaler'):
    # Update the plot
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))

        ax1.plot(train_loss, label='Train Loss')
        ax1.plot(test_loss, label='Test Loss')
        ax1.legend(loc='upper right')
        ax1.scatter(range(len(test_loss)), test_loss, color='red', label='Test Loss')
        ax1.scatter(range(len(accuracy)), train_loss, color='blue', label='Train Loss')
        ax1.set_title('%s Epoch: %d' % (optim_name,iteration+1))

        ax2.plot(accuracy, label='Accuracy')
        ax2.scatter(range(len(accuracy)), accuracy, color='blue', label='Accuracy')
        ax2.legend(loc='upper right')
        ax2.set_title('Accuracy: %.4f' % accuracy[-1])

        plt.tight_layout()
        plt.show()

def  main():
    if __name__ == '__main__':
        # Create some data
        train_loss = np.random.rand(100)
        test_loss = np.random.rand(100)
        accuracy = np.random.rand(100)

        # Plot the data
        loss_graph(train_loss, test_loss, accuracy, 'Geoadaler', 100)