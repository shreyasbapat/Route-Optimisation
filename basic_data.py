import requests
import json
import pandas as pd

def input_data():
    """overwrites file given in variable 'file_path' with data in required format
    Data files:
        'data/passengers.txt': List of boarding points, each corresponding to one passenger
        'data/boarding_times.csv': List of distinct boarding points with boarding time
    due date same for all points as "735"
    service time same for all points as "90"
    """
    no_vehicles = 25
    capacity = 200
    max_dist = 1000

    file_path = 'c102.txt'

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
    df = df.set_index("stop", drop = False)

    with open('apikey', 'r') as f: 
        api_key = f.read().strip()

    url = 'https://maps.googleapis.com/maps/api/geocode/json'

    with open(file_path, 'w') as f:
        params = {
            'key': api_key,
            'address': 'Bosch Bidadi, Bangalore'
        }

        r = requests.get(url, params=params)
        ans = r.json()

        lat = ans['results'][0]['geometry']['location']['lat']
        lng = ans['results'][0]['geometry']['location']['lng']

        f.write("C101 \n\nVEHICLE \nNUMBER     CAPACITY    KMS/TRIP\n" + str(no_vehicles) + "        " + str(capacity) + "         " + str(max_dist) + " \n\nCUSTOMER \nCUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME\n\n")
    
        f.write('   ' + '0      ' + str(lat) + '      ' + str(lng) + '      ' + '0' + '      ' + '0' + '      ' + '735' + '      ' + '0' + '\n')        
    
        for point in df['stop']:
            if point in d:
                params = {
                    'key': api_key,
                    'address': point+', Bangalore'
                }

                r = requests.get(url, params=params)
                ans = r.json()

                lat = ans['results'][0]['geometry']['location']['lat']
                lng = ans['results'][0]['geometry']['location']['lng']

                f.write('   ' + str(count) + '      ' + str(lat) + '      ' + "{0:.7f}".format(lng) + '      ' + str(d[point]) + '      ' + str(df.loc[point, 'time']) + '      ' + '735' + '      ' + '90' + '\n')
                count += 1
