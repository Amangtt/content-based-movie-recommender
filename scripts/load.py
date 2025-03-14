import pandas as pd
from numpy import genfromtxt
import pickle as pickle
import tabulate
from collections import defaultdict
import csv
import numpy as np
def load_data():
   
    movie_dict = defaultdict(dict)
    count = 0
#    with open('./data/movies.csv', newline='') as csvfile:
    with open('/Users/amantebeje/Desktop/projects/movie-recommender-/Data/content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1  #skip header
                #print(line) print
            else:
                count += 1
                movie_id = int(line[0])
                movie_dict[movie_id]["title"] = line[1]
                movie_dict[movie_id]["genres"] = line[2]

    
    
    return movie_dict

def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs

def print_pred_movies(y_p, item, movie_dict, maxcount=10):
    """ print results of prediction of a new user. inputs are expected to be in
        sorted order, unscaled. """
    count = 0
    disp = [["y_p", "movie id", "rating", "title", "genres"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        movie_id = item[i, 0].astype(int)
        disp.append([np.around(y_p[i, 0], 1), item[i, 0].astype(int), np.around(item[i, 2].astype(float), 1),
                     movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
    return table
def predictions_to_json(y_p, item, movie_dict, maxcount=10):
    """Convert predictions to JSON format with structured output."""
    predictions = []

    for i in range(min(maxcount, y_p.shape[0])):
        movie_id = int(item[i, 0])  # Convert NumPy int to Python int
        predicted_rating = float(np.around(y_p[i, 0], 1))  # Convert NumPy float to Python float
        actual_rating = float(np.around(item[i, 2], 1))  # Same conversion

        # Ensure movie ID exists in dictionary
        movie_info = movie_dict.get(movie_id, {"title": "Unknown", "genres": "Unknown"})

        predictions.append({
            "movie_id": movie_id,
            "actual_rating": actual_rating,
            "title": movie_info["title"],
            "genres": movie_info["genres"]
        })

    return {"predictions": predictions}


