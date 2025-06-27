import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.show()

def plot_decision_boundary(model, X, y):
    x_min = min([i[0] for i in X]) - 0.5
    x_max = max([i[0] for i in X]) + 0.5
    y_min = min([i[1] for i in X]) - 0.5
    y_max = max([i[1] for i in X]) + 0.5
    h = 0.01
    xx = []
    yy = []
    grid = []
    x = x_min
    while x <= x_max:
        y_row = []
        row = []
        y = y_min
        while y <= y_max:
            row.append([x, y])
            y += h
        grid.append(row)
        xx.append(x)
        x += h

    zz = []
    for row in grid:
        zz.append([model.predict(point) for point in row])

    plt.contourf(xx, [y_min + h * i for i in range(len(zz[0]))], list(zip(*zz)), alpha=0.6)
    plt.scatter([i[0] for i in X], [i[1] for i in X], c=y, edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()
