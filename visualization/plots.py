import matplotlib.pyplot as plt


def plot_results(results):
    names = list(results.keys())
    scores = list(results.values())

    plt.barh(names, scores)
    plt.xlabel("R2 Score")
    plt.title("Model Comparison")
    plt.show()
