# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:24:17 2017

@author: admin
"""

import urllib2
# If you are using Python 3+, import urllib instead of urllib2

import json 


data =  {

        "Inputs": {

                "input1":
                {
                    "ColumnNames": ["Year", "Month", "DayofMonth", "DayOfWeek", "Carrier", "OriginAirportID", "DestAirportID", "CRSDepTime", "DepDelay", "DepDel15", "CRSArrTime", "ArrDelay", "ArrDel15", "Cancelled"],
                    "Values": [ [ "0", "0", "0", "0", "value", "0", "0", "0", "0", "0", "0", "0", "0", "0" ], [ "0", "0", "0", "0", "value", "0", "0", "0", "0", "0", "0", "0", "0", "0" ], ]
                },        },
            "GlobalParameters": {
}
    }

body = str.encode(json.dumps(data))

url = 'https://ussouthcentral.services.azureml.net/workspaces/34cba1bd56384ceaa15eb81ebe2a596b/services/139cf3256f2640a588f9ccd4af17bfa0/execute?api-version=2.0&details=true'
api_key = 'PluABiXx1NIaq90UkpHtNKAIl8/lrgCBgA2PwWmBb9a2DPncQ+lTD3cFkP3sQvAiju52Dvc6rb14DqIwK67vww==' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib2.Request(url, body, headers) 

try:
    response = urllib2.urlopen(req)

    # If you are using Python 3+, replace urllib2 with urllib.request in the above code:
    # req = urllib.request.Request(url, body, headers) 
    # response = urllib.request.urlopen(req)

    result = response.read()
    print(result) 
except urllib2.HTTPError, error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())

    print(json.loads(error.read()))                 