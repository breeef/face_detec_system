# coding=utf-8
import redis
import time
import os
import sys
import datetime

ip='127.0.0.1'

r = redis.Redis(host=ip, port=6379, db=0)



def store_board_time(name,ext):
    if not r.keys(name):
        print('------------insert-------',name)
        r.set(name, "time",ex=ext)
        return 1
    else:
        print('--------------------------------------------------',name)
        return 0





def search(name):
    #print(name_search)
    if not r.keys(name):
        print('noooooooooooooooooooooot---------------')
        return 0
    else:
        print('geeeeeeeeeeeeeeeeeeeeeeet')
        return 1



