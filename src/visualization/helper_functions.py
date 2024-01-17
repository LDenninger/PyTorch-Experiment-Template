import matplotlib.pyplot as plt
import numpy as np


def pyplot_to_numpy(fig):
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    return img_array