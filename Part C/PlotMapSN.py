"""
Plots a map of the San NiccolÃ² building
"""
import matplotlib.pyplot as plt


def PlotMapSN(Obstacles):
    # function that plots map with polygonal obstacles
    for i in range(len(Obstacles)):
        plt.fill(Obstacles[i][:, 0], Obstacles[i][:, 1], facecolor='lightgrey', edgecolor='black')



