from tornado.ioloop import IOLoop, PeriodicCallback
from tornado import gen
from tornado.websocket import websocket_connect

import numpy as np
import h5py

import json
import random
import string

import sys

import random

class Client(object):
    def __init__(self, url, timeout):
        self.f = h5py.File('pollen_scraped.h5', 'a')
        self.day = np.zeros((1, 24), dtype=np.int32)
        self.dataset = np.zeros((0, 24), dtype=np.int32)

        self.stations = ['albacete', 'alcazar', 'alicante', 'almeria', 'avila', 'badajoz',
         'barcelona', 'barcelona-uab', 'bejar', 'bilbao', 'burgos', 'burjassot', 'caceres',
         'cadiz', 'cartagena', 'castellon-de-la-plana', 'ciudad-real', 'cordoba', 'coruña',
         'cuenca', 'elche', 'gerona', 'granada', 'gijon', 'guadalajara', 'huelva', 'huesca',
         'jaen-hospital', 'jaen', 'jativa', 'las-palmas', 'leon', 'lerida', 'logroño',
          'madrid-subiza', 'madrid-hospital', 'malaga', 'murcia', 'oviedo', 'palencia',
          'palma-mallorca', 'pamplona', 'ponferrada', 'pontevedra', 'salamanca', 'san-sebastian',
          'santa-cruz-tenerife', 'santander', 'santiago-compostela', 'segovia', 'sevilla-macarena',
          'sevilla-tomillar', 'soria', 'talavera', 'tarragona', 'teruel', 'toledo', 'torrelavega',
           'tudela', 'valencia', 'valladolid', 'vitoria', 'zamora', 'zaragoza']


        self.ids = [1, 2, 3, 4, 5, 7, 6, 50, 8, 9, 11, 10, 12, 58, 16, 13, 15, 62, 14, 17, 18,
        19, 65, 20, 21, 60, 59, 23, 22, 24, 28, 25, 26, 27, 29, 57, 30, 31, 32, 33, 35, 34, 36,
        37, 38, 43, 45, 39, 64, 40, 41, 47, 42, 61, 44, 46, 48, 49, 63, 51, 54, 53, 55, 56]

        self.s = 5
        self.counter = 0

        self.url = url
        self.timeout = timeout
        self.ioloop = IOLoop.instance()
        self.ws = None
        self.connect()
        self.ioloop.start()

    @gen.coroutine
    def connect(self):
        print("trying to connect")
        try:
            self.ws = yield websocket_connect(self.url)
        except Exception(e):
            print("connection error")
        else:
            print("connected")
            self.run()

    @gen.coroutine
    def run(self):
        while True:
            payload = yield self.ws.read_message()
            if payload is None:
                print("connection closed")
                self.ws = None
                break

            if payload[0] == "o":
                msg = json.dumps(['{"msg":"connect", "version":"1", "support":["1","pre2","pre1"]}'])
                self.ws.write_message(msg.encode('utf8'))
            elif payload[0] == 'a':
                frame = json.loads(payload[1:])
                body = json.loads(frame[0])

                if 'msg' in body:
                    if body['msg'] == 'connected':
                        self.day = np.zeros((1, 24), dtype=np.int32)
                        self.dataset = np.zeros((0, 24), dtype=np.int32)
                        self.counter = 0

                        print('scraping from', self.stations[self.s])

                        stringmsg = '{{"msg":"sub","id":"die7LBp7mEbzW3ffQ","name":"polenes2015","features":[{{"selector":{{"$and":[{{"fecha":{{"$gte":19700101,"$lte":20201231}}}},{{"idEstacion":{}}}]}},"options":{{"sort":{{"fecha":1}}}},"jump":1}}]}}'.format(self.ids[self.s]);
                        #print(stringmsg)
                        msg = json.dumps([stringmsg])
                        self.ws.write_message(msg.encode('utf8'));
                    elif body['msg'] == 'ping':
                        print('pong')
                        msg = json.dumps(['{"msg": "pong"}'])
                        self.ws.write_message(msg.encode('utf8'));
                    elif body['msg'] == 'ready':
                        print(self.dataset.shape)

                        if self.dataset.shape[0] > 0:
                            print(self.dataset[0, 0], '-', self.dataset[-1, 0])

                            if self.stations[self.s] in self.f:
                                del self.f[self.stations[self.s]]
                            self.f.create_dataset(self.stations[self.s], data=self.dataset)
                        print('------------')

                        self.s += 1
                        if self.s < len(self.stations):
                            self.day = np.zeros((1, 24), dtype=np.int32)
                            self.dataset = np.zeros((0, 24), dtype=np.int32)
                            self.counter = 0

                            stringmsg = "{\"msg\":\"unsub\",\"id\":\"die7LBp7mEbzW3ffQ\"}"
                            msg = json.dumps([stringmsg])
                            self.ws.write_message(msg.encode('utf8'));

                            print('scraping from', self.stations[self.s])

                            stringmsg = '{{"msg":"sub","id":"die7LBp7mEbzW3ffQ","name":"polenes2015","features":[{{"selector":{{"$and":[{{"fecha":{{"$gte":19700101,"$lte":20201231}}}},{{"idEstacion":{}}}]}},"options":{{"sort":{{"fecha":1}}}},"jump":1}}]}}'.format(self.ids[self.s]);
                            #print(stringmsg)
                            msg = json.dumps([stringmsg])
                            self.ws.write_message(msg.encode('utf8'));

                        else:
                            sys.exit()
                    elif body['msg'] == 'added':
                        self.day[0][0] = body['fields']['fecha'];

                        self.counter += 1
                        if self.counter % 365 == 0:
                            print(self.counter//365, 'years scraped')

                        for i in range(1, 23):
                            if str(i) in body['fields'] and isinstance(body['fields'][str(i)], (int, float)):
                                self.day[0][i] = body['fields'][str(i)]
                            else:
                                if str(i) in body['fields']:
                                    print('type error:', body['fields'][str(i)])
                                self.day[0][i] = 0;
                        self.dataset = np.append(self.dataset, self.day, axis = 0);


if __name__ == "__main__":
    client = Client("wss://www.polenes.com/sockjs/490/j6ieo28e/websocket", 5)
