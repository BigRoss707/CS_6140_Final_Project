import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import plotly.express as px
import plotly.io as pio

palette = sns.color_palette("bright", 10)

# Some of this code modified from https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis
def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    return (colour[0], colour[1], colour[2], alpha)

def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'predicted_cluster', color=palette)

    # Stagger the axes
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.predicted_cluster == i])

    # Draw the chart
    pc = px.parallel_coordinates(data_frame=df, color='predicted_cluster')
    pio.renderers.default = 'browser'
    pc.show()

    # Stagger the axes
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Draw the chart
    pc = px.parallel_coordinates(data_frame=df, color='predicted_cluster')
    pio.renderers.default = 'browser'
    pc.show()

    # Stagger the axes
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)
