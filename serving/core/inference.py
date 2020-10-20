# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QDNetInference
   Author :       machinelp
   Date :         2020-08-10
-------------------------------------------------

'''

import os
import sys
import json
import argparse
from utils.logging import logging
from flask_restful import Resource,Api
from core.models import QDNetModel
from qdnet.conf.config import load_yaml
from flask import Flask,abort, make_response, request, jsonify

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
parser.add_argument('--fold', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)

qdnet_model = QDNetModel(config, args.fold)

class QDNetInference(Resource):
    def __init__(self):
        pass

    def post(self):

        if not request.json or 'content' not in request.json :
            res = { "code": "400", "data": {}, "message": "request is not json or content not in json" }
            return jsonify ( res )

        else:
            logging.info( "[QDNetInference] [post] request.json:{}".format( request.json ) )
            url = request.json["content"]
            logging.info( "[QDNetInference] [post] url:{}".format( url ) )
            data = download(url)
            pre = qdnet_model.predict(data)
            res = { "code": "200", "data": pre, "message": "" }
            return jsonify ( res )
