import matplotlib.pyplot as plt
def draw_loss(train_losses,valid_losses,train_accs,valid_accs,epoch):
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.legend(['train', 'test'])
    plt.title('Model loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.show()

    plt.plot(train_accs)
    plt.plot(valid_accs)
    plt.legend(['train_acc', 'test_acc'])
    plt.title('Model acc', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Acc', fontsize=15)
    plt.show()
