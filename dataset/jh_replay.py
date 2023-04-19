#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla

import argparse

import os
import math
import time
import numpy as np
import pandas as pd
from PIL import Image

col_list = []


def col_callback(colli):
    global col_list
    print("Collision detected:\n" + str(colli) + "\n")
    print(str(colli.frame), str(colli.timestamp), str(colli.transform))
    print(str(colli.actor), str(colli.other_actor), str(colli.normal_impulse))
    impulse = colli.normal_impulse
    intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    col_list.append([colli.frame, intensity])


def dvs_cvt(image):
    dvs_events = np.frombuffer(
        image.raw_data,
        dtype=np.dtype(
            [
                ("x", np.uint16),
                ("y", np.uint16),
                ("t", np.int64),
                ("pol", np.bool),
            ]
        ),
    )
    dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    # Blue is positive, red is negative
    dvs_img[dvs_events[:]["y"], dvs_events[:]["x"], dvs_events[:]["pol"] * 2] = 255
    return dvs_img


def opt_cvt(image):
    image = image.get_color_coded_flow()
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "-s",
        "--start",
        metavar="S",
        default=0.0,
        type=float,
        help="starting time (default: 0.0)",
    )
    argparser.add_argument(
        "-d",
        "--duration",
        metavar="D",
        default=0.0,
        type=float,
        help="duration (default: 0.0)",
    )
    argparser.add_argument(
        "-f",
        "--recorder-filename",
        metavar="F",
        default="accident0418_1.log",
        help="recorder filename (accident1.log)",
    )
    argparser.add_argument(
        "-c",
        "--camera",
        metavar="C",
        default=0,
        type=int,
        help="camera follows an actor (ex: 82)",
    )
    argparser.add_argument(
        "-x",
        "--time-factor",
        metavar="X",
        default=1.0,
        type=float,
        help="time factor (default 1.0)",
    )
    argparser.add_argument(
        "-i", "--ignore-hero", action="store_true", help="ignore hero vehicles"
    )
    argparser.add_argument(
        "--spawn-sensors",
        action="store_true",
        help="spawn sensors in the replayed world",
    )
    args = argparser.parse_args()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)

        # set the time factor for the replayer
        client.set_replayer_time_factor(args.time_factor)

        # set to ignore the hero vehicles or not
        client.set_replayer_ignore_hero(args.ignore_hero)
        world = client.get_world()

        # --------------
        # Change weather conditions
        # --------------

        # weather = world.get_weather()
        # weather.sun_altitude_angle = -30
        # weather.fog_density = 65
        # weather.fog_distance = 10
        # world.set_weather(weather)

        # replay the session
        print(
            client.replay_file(
                args.recorder_filename,
                args.start,
                args.duration,
                args.camera,
                args.spawn_sensors,
            )
        )
        print(args.spawn_sensors)
        print(world.get_actors().filter("*sensor*"))
        # print(world.get_actors())
        actor_list = world.get_actors().filter("*vehicle*")
        for actor in actor_list:
            if actor.attributes["role_name"] == "hero":
                args.camera = actor.id
                print("hero id is", args.camera)
        folder_name = args.recorder_filename.split(".")[0]
        ## added
        ego_vehicle = world.get_actor(args.camera)
        ego_col = world.get_actor(args.camera + 1)
        ego_cam_rgb = world.get_actor(args.camera + 2)
        ego_cam_depth = world.get_actor(args.camera + 3)
        ego_cam_depth02 = world.get_actor(args.camera + 4)
        ego_cam_semseg = world.get_actor(args.camera + 5)
        ego_cam_dvs = world.get_actor(args.camera + 6)
        ego_cam_lidar = world.get_actor(args.camera + 7)
        ego_cam_optical_flow = world.get_actor(args.camera + 8)

        c_raw = carla.ColorConverter.Raw
        c_log_depth = carla.ColorConverter.LogarithmicDepth
        c_depth = carla.ColorConverter.Depth
        c_palette = carla.ColorConverter.CityScapesPalette

        # time.sleep(0.5)
        ego_col.listen(lambda colli: col_callback(colli))
        ego_cam_rgb.listen(
            lambda image: image.save_to_disk(
                "{}/rgb/{:06d}.png".format(folder_name, image.frame), c_raw
            )
        )
        ego_cam_depth.listen(
            lambda image: image.save_to_disk(
                "{}/log_depth/{:06d}.png".format(folder_name, image.frame), c_log_depth
            )
        )
        # ego_cam_depth02.listen(
        #     lambda image: image.save_to_disk(
        #         "{}/raw_depth/{:06d}.png".format(folder_name, image.frame), c_depth
        #     )
        # )
        ego_cam_semseg.listen(
            lambda image: image.save_to_disk(
                "{}/semseg/{:06d}.png".format(folder_name, image.frame), c_palette
            )
        )
        if not os.path.exists("{}/dvs".format(folder_name)):
            os.makedirs("{}/dvs".format(folder_name))
        ego_cam_dvs.listen(
            lambda image: Image.fromarray(dvs_cvt(image)).save(
                "{}/dvs/{:06d}.png".format(folder_name, image.frame)
            )
        )

        # ego_cam_lidar.listen(
        #     lambda point_cloud: point_cloud.save_to_disk(
        #         "{}/lidar/{:06d}.ply".format(folder_name, point_cloud.frame)
        #     )
        # )

        # if not os.path.exists("{}/optical_flow".format(folder_name)):
        #     os.makedirs("{}/optical_flow".format(folder_name))
        # ego_cam_optical_flow.listen(
        #     lambda image: Image.fromarray(opt_cvt(image)).save(
        #         "{}/optical_flow/{:06d}.png".format(folder_name, image.frame)
        #     )
        # )

        while True:
            world_snapshot = world.wait_for_tick()
            print(world_snapshot)

    finally:
        print(col_list)
        df = pd.DataFrame(data=col_list, columns=["frame", "intensity"])
        df.to_csv("{}/collision.csv".format(folder_name), index=False)
        print("Collision information saved.")
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")
