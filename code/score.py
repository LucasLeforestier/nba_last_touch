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

def get_video(period: int, clocktime: str, time_elapsed: int, link: str):
    
    """
    Get the video from the JSONL

    period (int): number of the period (1 to 4)
    clocktime (str): time left in the period
    time_elapsed (int): number of seconds, decimals are accepted
    link (str): path to the file (JSONL)
    """

    data = []
    raw_data = []
    with jsonlines.open(link) as f:
        sequence = True
        for line in f.iter():
            data_time = line["payload"]["time"]

            ## if sequence == True then the sequence wanted has not started yet
            if sequence:
                if (data_time["period"] == period) and (str_to_time(data_time["gameClockTime"]) <= (str_to_time(clocktime))):
                    get_players = pd.DataFrame(line["payload"]["details"]["players"])
                    starting_time = get_timestamp_str(data_time["timeUTC"])
                    data.append([line["payload"]["sequences"]["frame"], data_time, line["payload"]["summary"], line["payload"]["samples"]])
                    raw_data.append(line)
                    sequence = False
                    ## add everything (first frame)

            ## Break the for loop whenever the time elapsed is done
            elif (not sequence) and (data_time["period"] == period) and (str_to_time(data_time["gameClockTime"]) <= (str_to_time(clocktime))) and (get_timestamp_str(data_time["timeUTC"]) <= (starting_time + timedelta(milliseconds=time_elapsed*1000))):
                data.append([line["payload"]["sequences"]["frame"], data_time, line["payload"]["summary"], line["payload"]["samples"]])
                raw_data.append(line)
            else:
                break

    print(f"Size of the data is {len(data)} frames")

    ## OUTPUTS
    # data: formated to get the score of closeness
    # get_players: list of all players
    # raw_data: the data as it is, to calculate the score of similarity


    return data, get_players, raw_data


def get_score_velocity(data) :

    """
    Outputting the score of velocity in a given sequence
    """


    # Fetch ball trajectory
    time, x_ball, y_ball, z_ball = get_ball_trajectory(data)

    # Get ball velocity
    dxdt, _ = get_derivatives(x_ball, time, 15, 3)
    dydt, _ = get_derivatives(y_ball, time, 15, 3)
    dzdt, _ = get_derivatives(z_ball, time, 15, 3)
    ball_velocity = np.vstack((dxdt, dydt, dzdt)).T

    #ball_pos = np.vstack((x_ball, y_ball, z_ball)).T
    #ball_acceleration = np.vstack((dxdt2, dydt2, dzdt2)).T

    # Store player IDs (!IMPORTANT: assumes no substitution during the sequence)
    ids = [x.get('playerId').get('nbaId') for x in data[0].get('payload').get('samples').get('people')]
    # Create dictionery to store player hands trajectories
    hand_traj = dict(zip(ids, [{'left': {'x': [], 'y': [], 'z': []}, 'right': {'x': [], 'y': [], 'z': []}}  for _ in range(len(ids))]))

    # Loop through the frames to populate the hand trajectories
    for line in data :
        for player in line.get('payload').get('samples').get('people') :

            player_id = player.get('playerId').get('nbaId')

            if player_id not in hand_traj.keys() :
                print('Substitution !!!')
            
            xl, yl, zl = (np.array(player.get('joints')[0].get('lPinky')) + np.array(player.get('joints')[0].get('lThumb'))) / 2
            xr, yr, zr = (np.array(player.get('joints')[0].get('rPinky')) + np.array(player.get('joints')[0].get('rThumb'))) / 2

            # Left-hand trajectory
            hand_traj.get(player_id).get('left').get('x').append(xl)
            hand_traj.get(player_id).get('left').get('y').append(yl)
            hand_traj.get(player_id).get('left').get('z').append(zl)
            # Right-hand trajectory
            hand_traj.get(player_id).get('right').get('x').append(xr)
            hand_traj.get(player_id).get('right').get('y').append(yr)
            hand_traj.get(player_id).get('right').get('z').append(zr)


    # Initiate dictionary to store hand-ball velocity similarities
    velsimilarity = dict(zip(ids, [{'left': [], 'right': []}  for _ in range(len(ids))]))

    # Loop through the hand trajectories dictionary
    for player_id, hand in hand_traj.items() :

        # Compute left-hand velocity
        xl_hand_vel, _ = get_derivatives(hand.get('left').get('x'), time, 15, 3)
        yl_hand_vel, _ = get_derivatives(hand.get('left').get('y'), time, 15, 3)
        zl_hand_vel, _ = get_derivatives(hand.get('left').get('z'), time, 15, 3)
        left_hand_velocity = np.vstack((xl_hand_vel, yl_hand_vel, zl_hand_vel)).T

        # Compute right-hand velocity
        xr_hand_vel, _ = get_derivatives(hand.get('right').get('x'), time, 15, 3)
        yr_hand_vel, _ = get_derivatives(hand.get('right').get('y'), time, 15, 3)
        zr_hand_vel, _ = get_derivatives(hand.get('right').get('z'), time, 15, 3)
        right_hand_velocity = np.vstack((xr_hand_vel, yr_hand_vel, zr_hand_vel)).T

        # Loop through the frames
        for b_vel, lh_vel, rh_vel in zip(ball_velocity, left_hand_velocity, right_hand_velocity) :

            # Normalize velocity vectors
            b_vel_norm = b_vel / np.linalg.norm(b_vel)
            lh_vel_norm = lh_vel / np.linalg.norm(lh_vel)
            rh_vel_norm = rh_vel / np.linalg.norm(rh_vel)
            
            # Append list of similarity to dictionary
            velsimilarity.get(player_id).get('left').append((2 - np.linalg.norm(b_vel_norm - lh_vel_norm)) / 2)
            velsimilarity.get(player_id).get('right').append((2 - np.linalg.norm(b_vel_norm - rh_vel_norm)) / 2)

    return velsimilarity


def compute_proba(score_clos, score_sim, times, factor = 0.5, factor_sim = 1):
    
    """
    Displays the final proba in a given sequence

    score_clos (dict): the score of closeness of the sequence
    score_sim (dict): the score of velocity simlarity of the sequence (reshaped)
    factor (float): parameters of decay, factor that multiply the importance of the previous frame compared to the one following
    factor_sim (float): parameters of velocity imortance, look for engineering report to understand
    """
    
    ## setting up the plot
    fig, axs = plt.subplots(4, 1, figsize=(14, 20))
    areas = {}
    colors = {}
    col_ind = 0

    ## loop on the every frame
    for key, value in score_clos.items():
        if all(element == 0 for element in value):
            pass
        else:
            ## compute the scores and add every plot
            colors[key] = random_color()
            col_ind += 1
            areas[key] = area_factored_similarity(value, score_sim[key], times, factor = factor, factor_sim = factor_sim)
            axs[0].plot(times, value, color = colors[key], label = key, marker = ".")
            axs[1].plot(times, score_sim[key], color = colors[key], label = key, marker = "_")
            axs[2].plot(times[:-1], areas[key], color = colors[key], label = key, marker = "_")
            axs[2].fill_between(times[:-1],  areas[key], color = colors[key], alpha=0.3, label = f"Score: {round(np.sum(areas[key]), 5)}")
    
    ## score of closeness plot
    axs[0].set_ylabel("Score of closeness", fontsize = 20)
    axs[0].set_xlabel("Time", fontsize = 20)
    axs[0].legend(fontsize = 20, edgecolor = "w")
    axs[0].tick_params(labelsize=15)

    ## score of similarity plot
    axs[1].set_ylabel("Score of Velocity Similarity", fontsize = 20)
    axs[1].set_xlabel("Time", fontsize = 20)
    axs[1].legend(fontsize = 20, edgecolor = "w")
    axs[1].tick_params(labelsize=15)

    ## final score plot
    axs[2].set_ylabel("Area Score", fontsize = 20)
    axs[2].set_xlabel("Time", fontsize = 20)
    axs[2].legend(fontsize = 20, edgecolor = "w")
    axs[2].tick_params(labelsize=15)

    for key, value in areas.items():
        areas[key] = np.sum(areas[key])

    ## computing proba with a barplot
    proba = make_proba(areas)
    last_touch = max(areas, key=lambda k: areas[k])
    axs[3].bar(proba.keys(), proba.values(), color= colors.values())
    axs[3].set_xlabel('Players', fontsize = 20)
    axs[3].set_ylabel('Probability', fontsize = 20)
    axs[3].tick_params(labelsize=15)
    #axs[3].set_title(f'Who touched the ball last: {last_touch}', size = 15)
    axs[3].set_ylim(0, 1.1)
    for i, value in enumerate(proba.values()):
        axs[3].text(i, value + 0.02, f"{round(value, 3)*100}%", ha='center', size = 15)

    plt.tight_layout()
    plt.show()  
    