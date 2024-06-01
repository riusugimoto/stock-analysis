import os
import matplotlib.pyplot as plt

def plot_sp500(sp500):
    images_folder = r'C:\Users\agoo1\OneDrive\Documents\2024 summer\Data\mlstock\images'
    os.makedirs(images_folder, exist_ok=True)
    plot_path = os.path.join(images_folder, "sp500_plot.png")
    
    sp500.plot.line(y="Close", use_index=True)
    plt.savefig(plot_path)
    plt.close()

def plot_combined(combined):
    images_folder = r'C:\Users\agoo1\OneDrive\Documents\2024 summer\Data\mlstock\images'
    os.makedirs(images_folder, exist_ok=True)
    plot_path = os.path.join(images_folder, "sp500_plot.png")
    combined.plot()
    plt.savefig(plot_path)
    plt.close()
