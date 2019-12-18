import requests
import json
import pandas as pd
import numpy as np


def input_data():
    """overwrites file given in variable 'file_path' with data in required format
    Data files:
        'data/passengers.txt': List of boarding points, each corresponding to one passenger
        'data/boarding_times.csv': List of distinct boarding points with boarding time

    creates dists.txt with distance matrix
    """
    no_vehicles = 25
    capacity = 200
    max_dist = 1000
    due_date = 835
    service_time = 90
    
    stops = list()

    file_path = "c102.txt"

    with open("data/passengers.txt", "r") as datafile:
        lines = list(datafile)
        d = dict()

        for l in lines:
            l = l.rstrip()
            if l in d:
                d[l] += 1
            else:
                d[l] = 1

    df = pd.read_csv("data/boarding_times.csv")
    count = 1
    df = df.set_index("stop", drop=False)

    with open("apikey", "r") as f:
        api_key = f.read().strip()

    url = "https://maps.googleapis.com/maps/api/geocode/json"

    with open(file_path, "w") as f:
        params = {"key": api_key, "address": "Bosch Bidadi, Bangalore"}

        r = requests.get(url, params=params)
        ans = r.json()

        lat = ans["results"][0]["geometry"]["location"]["lat"]
        lng = ans["results"][0]["geometry"]["location"]["lng"]

        stops.append(str(lat) + ',' + str(lng))

        f.write(
            "C102 \n\nVEHICLE \nNUMBER     CAPACITY    KMS/TRIP\n"
            + str(no_vehicles)
            + "        "
            + str(capacity)
            + "         "
            + str(max_dist)
            + " \n\nCUSTOMER \nCUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME\n\n"
        )

        f.write(
            "   "
            + "0      "
            + str(lat)
            + "      "
            + str(lng)
            + "      "
            + "0"
            + "      "
            + "0"
            + "      "
            + str(due_date)
            + "      "
            + "0"
            + "\n"
        )

        for point in df["stop"]:
            if point in d:
                params = {"key": api_key, "address": point + ", Bangalore"}

                r = requests.get(url, params=params)
                ans = r.json()

                lat = ans["results"][0]["geometry"]["location"]["lat"]
                lng = ans["results"][0]["geometry"]["location"]["lng"]

                stops.append(str(lat)+','+str(lng))

                f.write(
                    "   "
                    + str(count)
                    + "      "
                    + str(lat)
                    + "      "
                    + "{0:.7f}".format(lng)
                    + "      "
                    + str(d[point])
                    + "      "
                    + str(df.loc[point, "time"])
                    + "      "
                    + str(due_date)
                    + "      "
                    + str(service_time)
                    + "\n"
                )
                count += 1


    url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    stops_num = len(stops)
    node_dist_mat = np.zeros((stops_num, stops_num))
    for i in range(stops_num):
        curr_stop = stops[i]
        node_dist_mat[i][i] = 1e-8
        for j in range(i + 1, stops_num):
            next_stop = stops[j]

            params = {"key": api_key, "origins": curr_stop, "destinations": next_stop}
            r = requests.get(url, params=params)
            x = r.json()

            if x["rows"][0]["elements"][0]["status"] == "OK":
                node_dist_mat[i][j] = x["rows"][0]["elements"][0]["distance"]["value"] / 1000
            else: node_dist_mat[i][j] = 100000
            node_dist_mat[j][i] = node_dist_mat[i][j]

    with open('dists.txt', 'w') as f:
        for i in range(stops_num):
            for j in range(stops_num):
                f.write(str(node_dist_mat[i][j]) + "    ")
            f.write('\n')