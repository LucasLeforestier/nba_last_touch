from datetime import datetime, timedelta
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random
import imageio
import matplotlib.patches as mpatches
from collections import defaultdict
import cv2
from utils import *
from score import *


def save_plots_proba(data, raw_data, output_folder, get_players, set_lim = [(-570, 570), (-300, 300), (0, 200)], factor = 0.5, factor_sim = 0.1, error_add = 3):
    
    """
    Output the frame needed to maka video in a given folder with the proba in the legend

    data (list of frames tailored for get score closeness): the first output of the get_video
    raw_data (list of frames tailored for get score velocity): the third output of the get_video
    output_folder (path, does not have to exist): where to stock the frames of the future video
    get_players (list): the second output of get_video
    set_lim (list of tuples): limit of the court, can be changed if want to zoom on a specific part of the court
    factor (float): parameters of decay, factor that multiply the importance of the previous frame compared to the one following
    factor_sim (float): parameters of velocity imortance, look for engineering report to understand
    """
    
    ## create a folder if the output path lead to a non-existing folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' has been created.")

    ## plot all the lines of the basketball court
    out_lines = np.array([[-564, -300, 0], [-564, 300, 0], [564, 300, 0], [564, -300, 0]])
    links_lines = [(0, 1), (1, 2), (2, 3), (3, 0)]

    three_point_lines = np.array([[-564, -264, 0], [-402, -264, 0], [-564, 264, 0], [-402, 264, 0], [564, -264, 0], [402, -264, 0], [564, 264, 0], [402, 264, 0]])
    links_3plines = [(0, 1), (2, 3), (4, 5), (6, 7)]

    box_lines = np.array([[-564, -72, 0], [-342, -72, 0], [-564, 72, 0], [-342, 72, 0], [564, -72, 0], [342, -72, 0], [564, 72, 0], [342, 72, 0]])
    links_boxlines = [(0, 1), (2, 3), (1, 3), (4, 5), (6, 7), (5, 7)]

    basket_backboard = np.array([[-516, 31, 120], [-516, 31, 168], [-516, -31, 120], [-516, -31, 168], [516, 31, 120], [516, 31, 168], [516, -31, 120], [516, -31, 168]])
    links_backlines = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7)]

    middle_line = np.array([[0, -300, 0], [0, 300, 0]])
    links_middle = [(0, 1)]

    circle3pts1 = np.array([[circle_equation(y), y, 0] for y in np.linspace(-264, 264, 529)])
    circle3pts2 = np.array([[circle_equation(y, inv=False), y, 0] for y in np.linspace(-264, 264, 529)])

    x1, y1, z1 = get_circles(9, (-505, 0, 120))
    x2, y2, z2 = get_circles(9, (505, 0, 120))
    x3, y3, z3 = get_circles(72, (0, 0, 0))


    ## compute the two scores
    score_clos, stop_index, times = get_score_closeness(data, get_players, error_add= error_add)
    score_sim = get_score_velocity(raw_data)
    score_sim = reshape_score(score_sim, data, get_players, stop_index)

    ## get the id of the offense team, to distinguish them
    team_id_offense = data[0][2]["teamTouches"]["teamId"]["nbaId"]


    ## loop on every frame
    for k, data in enumerate(data):

        ## create dict to store the proba of players and their assigned color
        proba = {}
        color_proba = {}
        
        ## getting the frames
        fig = plt.figure(figsize = (16, 16))
        ax = fig.add_subplot(111, projection='3d')

        ## loop on people
        for l in range(len(data[3]["people"])):

            ## get the skeleton of the player at a frame, their id and name
            player, link = plot_skeleton(data,l)
            playerId = data[3]["people"][l]["playerId"]["nbaId"]
            playerName = get_name(get_players, playerId)

            ## assign a color wheter he's on offense or defense
            if get_nba_team(get_players, playerId) == team_id_offense:
                color = "g"
            else :
                color = "b"

            ## plot the skeleton with the right color, plus the links
            ax.scatter(player[:,0], player[:,1], player[:,2], c=color, marker='o', s= 50)
            for i, j in link:
                ax.plot([player[i,0], player[j,0]], [player[i,1], player[j,1]], [player[i,2], player[j,2]], c=color)

            ## compute his proba at a given frame
            proba[playerName] = np.sum(area_factored_similarity(score_clos[playerName][:k+1], score_sim[playerName][:k+1], times[:k+1], factor = factor, factor_sim = factor_sim))
            color_proba[playerName] = color

        ## computing the proba 
        proba = make_proba(proba)
        
        ## plot the ball
        ball = plot_ball(data)
        ax.scatter(ball[0], ball[1], ball[2], c='r', marker='o', s= 50)
        ax.set_title(f"Period {data[1]['period']}, Time : {data[1]['gameClockTime']}", fontsize = 20)


        ## run all the lines
        run_lines(out_lines, links_lines, ax)
        run_lines(three_point_lines, links_3plines, ax)
        run_lines(box_lines, links_boxlines, ax)
        run_lines(basket_backboard, links_backlines, ax)
        run_lines(middle_line, links_middle, ax)
        ax.plot(x1, y1, z1, c = 'k')
        ax.plot(x2, y2, z2, c = 'k')
        ax.plot(x3, y3, z3, c = 'k')
        
        ax.scatter(circle3pts1[:, 0], circle3pts1[:, 1], circle3pts1[:, 2], color='k', s = 1)
        ax.scatter(circle3pts2[:, 0], circle3pts2[:, 1], circle3pts2[:, 2], color='k', s = 1)
        
        if set_lim is not None:
            ax.set_xlim(set_lim[0])
            ax.set_ylim(set_lim[1])
            ax.set_zlim(set_lim[2])
        
        ## Get the legend with the proba
        ax.view_init(elev=45, azim=-70)
        to_plot = []
        for key, value in dict(sorted(proba.items(), key=lambda item: item[1], reverse=True)).items():
            if value > 0:
                to_plot.append(mpatches.Patch(color = color_proba[key], label=f'{key}: {str(round(value, 3)*100)[:4]}%'))
            else:
                break
        ax.legend(handles=to_plot, fontsize = 25)
        ax.set_title(f"Period {data[1]['period']}, Time : {data[1]['gameClockTime']}", fontsize = 25)
        ax.set_box_aspect([2, 1, 0.6])

        ax.grid(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_proj_type('ortho')

        ## Save all the frames
        if k <= 9:
            name = f'000{k}'
        
        elif k >= 10 and k <= 99:
            name = f'00{k}'
        elif k >= 100 and k <= 999:
            name = f'0{k}'
        else :
            name = k
        plt.savefig(os.path.join(output_folder, f'plot_{name}.png'))
        plt.close(fig)

    print("3D Plots saved in folder:", output_folder)


def create_video(path: str, output: str, fps: int):

    """
    Create a video from a series of frames 

    path (str): the path to the files with the frames
    output (str): path to get the output video
    fps (int): number of fps wanted
    """

    current_directory = os.getcwd()
    # Directory where your matplotlib images are stored
    image_folder = os.path.join(current_directory, path)

    # Get all image filenames in the directory
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # Sort the images based on their filenames
    images.sort()
    # Create a list to store the images
    image_list = []
    for filename in images:
        # Read each image and append to the list
        image = imageio.imread(os.path.join(image_folder, filename))
        image_list.append(image)

    # Output GIF filename
    mp4_filename = f'{output}.mp4'

    # Save the list of images as a GIF
    imageio.mimsave(mp4_filename, image_list, fps=fps)