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

def str_to_time(string):
    """
    Convert string to timestamp
    """
    try:
        return datetime.strptime(string, '%M:%S')
    except :
        return datetime.strptime(string, '%S.%f')
    
def get_timestamp_str(string):
    """
    Convert string into right format with what's given in the dataset
    """
    try :
        return datetime.strptime(string.split("T")[1], '%H:%M:%S.%f')
    except ValueError:
        try:
            return datetime.strptime(string.split("T")[1][:-1], '%H:%M:%S.%f')
        except ValueError:
            return datetime.strptime(string.split("T")[1][:-1], '%H:%M:%S')


def get_ball_handler(data):
    
    """
    Getting the ball handler from a frame
    """

    return data[2]["teamTouches"]["playerId"]["nbaId"]

def get_team_handler(data):

    """
    Getting the ball handler's team from a frame
    """

    return data[2]["teamTouches"]["teamId"]["nbaId"]

def get_player_number(players, player_id):

    """
    Getting the player's number from an id
    """

    for _, row in players.iterrows():
        if row.id["nbaId"] == player_id:
            return row.jerseyNumber

def identify_id(data, id):

    """
    Getting the player from their id
    """

    for i in range(len(data[3]["people"])):
        if data[3]["people"][i]["playerId"]["nbaId"] == id:
            return i

def get_nba_team(players, player_id):

    """
    Getting the player's team from their id
    """

    for _, row in players.iterrows():
        if row.id["nbaId"] == player_id:
            return row.teamId["nbaId"]
        
def get_name(players, player_id):

    """
    Getting the player's name from their id
    """

    for _, row in players.iterrows():
        if row.id["nbaId"] == player_id:
            return row.shortName

def plot_skeleton(data, player = 0, plot = False):

    """
    First function to plot the skeleton of a player at a given frame
    """


    ## Get all the joints positions
    joints_position = data[3]["people"][player]["joints"][0]
    lbigtoe_pos = joints_position['lBigToe'] #0
    lankle_pos = joints_position['lAnkle'] #1
    lknee_pos = joints_position['lKnee'] #2
    midhip_pos = joints_position['midHip'] #3
    rknee_pos = joints_position['rKnee'] #4
    rankle_pos = joints_position['rAnkle'] #5
    rbigtoe_pos = joints_position['rBigToe'] #6
    neck_pos = joints_position['neck'] #7
    lelbow_pos = joints_position['lElbow'] #8
    lwrist_pos = joints_position['lWrist'] #9
    relbow_pos = joints_position['rElbow'] #10
    rwrist_pos = joints_position['rWrist'] #11
    nose_pos = joints_position['nose'] #12
    skeleton = np.array([lbigtoe_pos, lankle_pos, lknee_pos, midhip_pos, rknee_pos, rankle_pos, rbigtoe_pos, neck_pos, lelbow_pos, lwrist_pos, relbow_pos, rwrist_pos, nose_pos])
    
    ## How to link them with one another
    parts_to_link = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7), (7, 8), (8, 9), (7, 10), (10, 11), (7, 12)]

    ## makes the plot if needed
    if plot:
        fig = plt.figure(figsize = (12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(skeleton[:,0], skeleton[:,1], skeleton[:,2], c='g', marker='o', s= 50)
        for i, j in parts_to_link:
            ax.plot([skeleton[i,0], skeleton[j,0]], [skeleton[i,1], skeleton[j,1]], [skeleton[i,2], skeleton[j,2]], c='g')
        plt.show()
    
    return skeleton, parts_to_link

def plot_centroid(data, player = 0):

    """
    Get the centroid of a player at a given frame
    """

    centroid_position = data[3]["people"][player]["centroid"][0]["pos"]
    return  centroid_position

def plot_ball(data):

    """
    Get the centroid of the ball at a given frame
    """

    return  data[3]["ball"][0]["pos"]

def distance(point1, point2):

    """
    Compute the distance between two frames
    """
    point1 = np.array(point1)
    point2 = np.array(point2)

    distance = np.linalg.norm(point2 - point1)
    
    return distance

def get_centroid(data, playerId):

    """
    Get the centroid of a player (with their ID) at a given frame
    """

    return np.array(data[3]["people"][identify_id(data, playerId)]["centroid"][0]["pos"])

def get_closer_defender(data, get_players):
    
    """
    Get the centroid of a player (with their ID) at a given frame
    """
    
    ball_handler = get_centroid(data, get_ball_handler(data))
    team_handler = get_team_handler(data)
    team_centroid = []
    for i in range(len(data[3]["people"])):
        playerId = data[3]["people"][i]["playerId"]["nbaId"]
        player_centroid = get_centroid(data, playerId)
        if get_nba_team(get_players, playerId) != team_handler:
            team_centroid.append((playerId, distance(ball_handler, player_centroid)))
    return min(team_centroid, key=lambda x: x[1])[0]

def circle_equation(y, inv = True):

    """
    Get the equation of the three point line (for plot), hard-coded
    """

    if inv:
        return ((-168)/69696)*(y**2) - 234
    else : 
        return ((168)/69696)*(y**2) + 234
    
def run_lines(lines, links, ax, deuxD = False):

    """
    Run the lines in a plot (court's delimitation)
    """

    if deuxD:
        ax.scatter(lines[:,0], lines[:,1], c='k', marker='o')
        for i, j in links:
            ax.plot([lines[i,0], lines[j,0]], [lines[i,1], lines[j,1]], c='k')
    else:
        ax.scatter(lines[:,0], lines[:,1], lines[:,2], c='k', marker='o')
        for i, j in links:
            ax.plot([lines[i,0], lines[j,0]], [lines[i,1], lines[j,1]], [lines[i,2], lines[j,2]], c='k')

def get_circles(radius, center, deuxD = False):

    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    if deuxD: 
        return x, y
    else :
        z = center[2] + np.zeros_like(theta)
        return x, y, z
    

def get_ball_trajectory(raw_data) :
    
    """
    Getting the ball trajectory from the raw data
    """
    
    # Store the trajectory of the ball
    time , x_ball, y_ball, z_ball = [], [], [], []
    for line in raw_data :
        x, y, z = line.get('payload').get('samples').get('ball')[0].get('pos')
        if len(line.get('payload').get('time').get('timeET')) > 26 :
            t = datetime.strptime(line.get('payload').get('time').get('timeET'), '%Y-%m-%dT%H:%M:%S.%f%z')
        else :
            t = datetime.strptime(line.get('payload').get('time').get('timeET'), '%Y-%m-%dT%H:%M:%S%z')
        x_ball.append(x)
        y_ball.append(y)
        z_ball.append(z)
        time.append(t)

    return time, x_ball, y_ball, z_ball


def get_derivatives(vector, time, window_size, n) :

    """
    Getting the derivatives (i.e. the velocity) at a given frame
    """

    # Set k value from window size
    k = int((window_size - 1) / 2)

    # Initiate empty derivatives list
    dydt, dydt2 = [], []
    for i in range(k, len(vector) - k) :

        # Store time relative to first time point of window
        t = [(ti - time[i-k]).total_seconds() for ti in time[i-k:i+(k+1)]]

        # Get polynomial coefficients
        coef = np.polyfit(t, vector[i-k:i+(k+1)], n)

        # Calculate first and second derivative at the frame
        d1 = sum([(n - i) * coef[i] * t[k] ** (n - i - 1) for i in range(n)])
        d2 = sum([(n - i) * (n - i - 1) * coef[i] * t[k] ** (n - i - 2) for i in range(n - 1)])

        dydt.append(d1)
        dydt2.append(d2)

    # Pad the values on the boundaries
    dydt = k * [np.nan] + dydt + k * [np.nan]
    dydt2 = k * [np.nan] + dydt2 + k * [np.nan]

    return dydt, dydt2

def random_color():
    """
    Generate a random color
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def calculate_fourth_point(point1, point2, point3, distance):

    """
    Calculate the point for the index of a player, from the point of its thumb, pinky and wrist
    """

    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    midpoint = (p1 + p2) / 2

    direction_vector = midpoint - p3
    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
    fourth_point = p3 + distance * direction_vector_normalized

    return fourth_point, midpoint

def get_timestamp_frame(frame):

    """
    Getting the timestamp of a frame
    """

    try :
        return datetime.strptime(frame[1]["timeUTC"].split("T")[1], '%H:%M:%S.%f')
    except ValueError:
        try : 
            return datetime.strptime(frame[1]["timeUTC"].split("T")[1][:-1], '%H:%M:%S.%f')
        except ValueError:
            return datetime.strptime(frame[1]["timeUTC"].split("T")[1][:-1], '%H:%M:%S')


def filtering(times, players_pos):

    """
    Getting the first moment when at least one player is close enough, 
    in order to discard any moment afterwards in the sequence, therefore not influencing the proba

    """

    combined_lists = zip(*[reversed(v) for v in players_pos.values()])
    for k, valeurs in enumerate(combined_lists):
        if not all(val == 0 for val in valeurs):
            break

    try : 
        stop_index = len(times) - k + 1
    except UnboundLocalError:
        stop_index = -1

    return stop_index

def get_positions(player, average_hand = 9.5):

    """
    Getting the positions of the hands' players, with the new points
    """

    joints = player["joints"][0]
    lThumb = joints["lThumb"]
    rThumb = joints["rThumb"]
    lPinky = joints["lPinky"]
    rPinky = joints["rPinky"]
    lWrist = joints["lWrist"]
    rWrist = joints["rWrist"]
    lIndex, lPalm = calculate_fourth_point(lThumb, lPinky, lWrist, average_hand)
    rIndex, rPalm = calculate_fourth_point(rThumb, rPinky, rWrist, average_hand)

    return [lThumb, lPinky, lWrist, lIndex, lPalm, rThumb, rPinky, rWrist, rIndex, rPalm]

def get_close_players(frame, player, ball_error, body_error, radius_ball, error_add):
    
    """
    Attributing a score for a player by taking the nearest points (from his hands) to the ball
    """

    positions = get_positions(player)
    ball = plot_ball(frame)

    minimum = np.inf

    ## getting the closest points of his both hands
    for part in positions:
        if distance(part, ball) < minimum:
            minimum =  distance(part, ball)

    ## computing the lower and upper bound in which a possible touch is happening, based on error of positions
    ub = radius_ball + ball_error + body_error + error_add
    lb = radius_ball - (ball_error + body_error + error_add)

    ## if not under upper bound, discard the score (put to 0)
    if minimum > ub:
        return 0
    
    ## if in between, the score is the distannce to the upper bound divided by the maximum score (therefore it stays between 0 and 1)
    elif minimum > lb:
        return (ub - minimum)/(ub - lb)
    
    ## if over attribute 1
    else:
        return (ub - lb)/(ub - lb)


def make_proba(dico):

    """
    Computing the proba from the scores
    """

    summing = sum(dico.values())
    return {key: value / summing for key, value in dico.items()} 

def get_score_closeness(full_data, get_players, factor = 0.5, ball_error = 0.08, body_error = 0.32, radius_ball = 4.75, error_add = 0.8):
    
    """
    Getting the score of closeness of every player in the sequence
    """

    times = []
    players_pos = defaultdict(list)

    ## looping on the sequence
    for data in full_data:
        people = data[3]["people"]
        times.append(get_timestamp_frame(data))
        for player in people:
            players_pos[get_name(get_players, player["playerId"]["nbaId"])].append(get_close_players(data, player, ball_error, body_error, radius_ball, error_add))

    ## discarding the moments when nobody is touching the ball
    times = [ (time - times[0]).total_seconds() for time in times]
    stop_index = filtering(times, players_pos)
    for key, _ in players_pos.items():
        players_pos[key] = players_pos[key][:stop_index]
    return players_pos, stop_index, times[:stop_index]


def get_closer_hand(frame, player_id):

    """
    Getting the closest hand ot of the two (for the score of velocity similarity)
    """
    
    people = frame[3]["people"]
    for player in people:
        if player["playerId"]["nbaId"] == player_id:
            break

    positions = get_positions(player)
    ball = plot_ball(frame)

    minimum = np.inf
    closer = "left"
    for k in range(len(positions)):
        if distance(positions[k], ball) < minimum:
            minimum =  distance([k], ball)
            if k <= 4:
                closer = "left"
            else:
                closer = "right"

    return closer

def reshape_score(score_sim, full_data, get_players, stop_index):

    """
    Reshape the score of velocity similarity so that it matches the same format as the one of closeness
    """

    res = defaultdict(list)

    for k, frame in enumerate(full_data):
        if k == stop_index:
            break
        for key, value in score_sim.items():
            closer = get_closer_hand(frame, key)
            if np.isnan(value[closer][k]):
                res[get_name(get_players, key)].append(0)
            else:
                res[get_name(get_players, key)].append(value[closer][k])

    return res

def area_factored_similarity(curve, similarity, time, factor =0.5, factor_sim = 1):

    """
    Given a curve of closeness (curve), and one of similarity (similarity), it computes the area under the curve
    """

    area = []
    size = len(time) - 2
    curve = np.array(curve)*(np.array(similarity) ** factor_sim)

    for i in range(len(curve) - 1):
        segment_x = time[i:i+2]
        segment_y = curve[i:i+2]

        largeur_segment = segment_x[1] - segment_x[0]
        hauteur_moyenne = np.mean(segment_y)

        aire_segment = largeur_segment * hauteur_moyenne
        aire_ponderee = aire_segment * (factor ** (size - i))

        area.append(aire_ponderee)

    return area
