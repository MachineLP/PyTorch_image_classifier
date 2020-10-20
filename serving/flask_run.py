# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  Flask
   Author :       machinelp
   Date :         2020-08-27
-------------------------------------------------

'''
from core.inference import QDNetInference
from flask_restful import Resource,Api
from flask import Flask,abort, make_response, request, jsonify


app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')
api.add_resource(QDNetInference, '/infer')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9939, debug=True)




