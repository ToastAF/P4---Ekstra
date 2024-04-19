import argparse
import librosa
import os
import numpy as np

from pythonosc import osc_message_builder
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

import methods
from interface import ImpactDrums

ip = "localhost"
port_listen = 5006
port_send = 5003


instance = ImpactDrums()

def newplane(key, data):
    angle_x, angle_y = data
    path = methods.new_plane(instance, angle_x, angle_y)
    client.send_message("/plane", path)

def newfile(key, data):
    filepath = data
    path = methods.new_file_received(instance, filepath)
    client.send_message("/file", path)

def newrange(key, data):
    anglemax = data
    path = methods.change_range(instance, anglemax)
    client.send_message("/range", path)

def newcenter(key, data):
    path = methods.new_center(instance)
    client.send_message("/center", path)

def recenter(key, data):
    path = methods.recenter(instance)
    client.send_message("/center", path)

def generate(key, *data):
    print(data)
    xy, angle_max, loudness = data
    angle_x, angle_y = xy.split()
    angle_x, angle_y = float(angle_x), float(angle_y)
    path = methods.generate_sound(instance, angle_x, angle_y, angle_max, loudness)
    client.send_message("/sound", path)



if __name__ == "__main__":
    # Enable send OSC messages on port 5005
    client = udp_client.SimpleUDPClient(ip, port_send)
    # Enable reception of OSC messages on port 5006
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/newplane", newplane)
    dispatcher.map("/newpath", newfile)
    dispatcher.map("/newrange", newrange)
    dispatcher.map("/newcenter", newcenter)
    dispatcher.map("/recenter", recenter)
    dispatcher.map("/generate", generate)

    server = osc_server.ThreadingOSCUDPServer((ip, port_listen), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
