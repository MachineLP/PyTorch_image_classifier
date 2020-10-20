import requests
import pandas as pd
import json
import time
import numpy as np
import librosa
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help="update mode", type=str, default="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    url = "http://127.0.0.1:9949/infer"
    print ( ">>>>>", args.text )
    json_data = json.dumps( {"content": args.url} )
    headers = {'content-type': 'application/json'}
    start_time = time.time()
    respond = requests.request("POST", url, data=json_data, headers=headers)
    print ("time>>>>>>>", time.time() - start_time)
    pre = np.array( respond.json()["data"] )
    print ("pre>>>>>>>", pre)

