import os
import matplotlib.pyplot as plt

def save_image(data, filename="som_output.png", folder="outputs"):
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    plt.imsave(full_path, data)
    return full_path
