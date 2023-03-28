import matplotlib.pyplot as plt
import numpy as np

# def plotLine(X:np.ndarray, Y_true:np.ndarray, Y_pred:np.ndarray):
#     plt.scatter(X, Y_true, color = 'black')
#     plt.plot(X, Y_pred, color = 'green', marker = 'o')
#     plt.show()

def plotLine(X:np.ndarray, Y_true:np.ndarray, a:float, b:float, title:str, color:str, loss:str): # poslat truth pravac ko param i nacrtat
    # x = np.linspace(-5,5,100)
    # y = a * X + b
    # a, b = f'{a.item():.2f}', f'{b.item():.2f}'
    # X, y = int(X), int(y)
    # plt.plot(X, Y_true, '-r', label='y=2x+1')
    plt.scatter(X, Y_true, color = 'blue', label='true value')
    # plt.plot(X.item(), y.item(), '-r', label = f'y = {a}x + {b}') # 3. '-r', = ----- red  #, color = 'green', marker = 'o'
    plt.axline((0, b), slope = a, linewidth = 4, color = color, label = f'y = {a:.2f}x + {b:.2f}')

    # y = 2 * X + b

    plt.title(f'{title}') #f'Graph of y = {a:.2f} x + {b:.2f}'
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.xkcd() #LoL
    # plt.figtext(10,0, loss)
    plt.legend()
    plt.grid()
    plt.show()

def plotLoss(epochs:int, losses:float, color:str, title:str):
    epochs_range = range(1, epochs + 2)

    plt.title(f'{title} Loss')

    plt.xlabel('epochs')
    plt.xticks(epochs_range)

    plt.ylabel('loss')

    plt.legend()
    plt.grid()

    for i in range(epochs + 1):
        plt.scatter(i + 1, losses[i], color = color)
    
    plt.plot(epochs_range, losses, color = color)
    plt.show()


def plotLosses(epochs:int, losses:float, color:str, title:str):
    epochs_range = range(1, epochs + 2)

    fig, axes = plt.subplots(2)

    axes.set_title(f'{title} Loss')

    axes.set_xlabel('epochs')

    axes.set_xticks(epochs_range)

    axes.set_ylabel('loss')

    axes.set_ylim(ymin = 1)

    axes.legend()
    axes.grid()

    for i in range(epochs + 1):
        axes.scatter(i + 1, losses[i], color = color)
    
    axes.plot(epochs_range, losses, color = color)

    fig.show() # plt.show(fig)