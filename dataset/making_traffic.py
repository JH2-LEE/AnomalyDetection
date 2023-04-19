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

import random
import time

import math
import numpy as np
from PIL import Image


def col(event):
    impulse = event.normal_impulse
    intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    return event.frame, intensity


def rgb_cvt(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


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


def main():
    try:
        actor_list = []
        sensor_list = []
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()  # default: Town 10

        print("Recording on file: %s" % client.start_recorder("recording01.log", True))

        spectator = world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=-10, y=30, z=180), carla.Rotation(pitch=-90)
            )
        )

        spawn_points = world.get_map().get_spawn_points()

        # Blueprint
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter("*vehicle*")
        sensor_bps = blueprint_library.filter("*sensor*")

        # Ego Vehicle
        # ego_vehicle_bp = random.choice(vehicle_bps)
        # transform = random.choice(spawn_points)
        # ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform)
        # actor_list.append(ego_vehicle)
        # print("created ego %s" % ego_vehicle.type_id)
        # ego_vehicle.set_autopilot(True)

        # Manually spawn vehicles
        # for ind in [89, 95, 99, 102, 103, 104, 110, 111, 115, 126, 135, 138, 139, 140, 141]:
        # world.try_spawn_actor(random.choice(vehicle_bps), spawn_points[ind])

        # Randomly spawn vehicles
        for ind in range(0, 30):
            vehicle = world.try_spawn_actor(
                random.choice(vehicle_bps), random.choice(spawn_points)
            )
            if vehicle is not None:
                actor_list.append(vehicle)
                print("created %s" % vehicle.type_id)
                vehicle.set_autopilot(True)

        # RGB Camera
        # fps = 30.0
        # tick = str(1.0 / fps)
        # camera_rgb_bp = blueprint_library.find("sensor.camera.rgb")
        # # camera_rgb_bp.set_attribute("sensor_tick", tick)
        # camera_rgb_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        # camera_rgb = world.spawn_actor(
        #     camera_rgb_bp, camera_rgb_transform, attach_to=ego_vehicle
        # )
        # sensor_list.append(camera_rgb)
        # print("created ego rgb cam %s" % camera_rgb.type_id)
        # c_raw = carla.ColorConverter.Raw
        # camera_rgb.listen(
        #     lambda image: image.save_to_disk("rgb/%06d.png" % image.frame, c_raw)
        # )

        # Depth Camera
        # camera_depth_bp = blueprint_library.find("sensor.camera.depth")
        # camera_depth_bp.set_attribute("sensor_tick", tick)
        # camera_depth_transform = carla.Transform(
        #     carla.Location(x=1.5, z=2.4)
        # )  # without calibration
        # camera_depth = world.spawn_actor(
        #     camera_depth_bp, camera_depth_transform, attach_to=ego_vehicle
        # )
        # sensor_list.append(camera_depth)
        # print("created ego depth cam %s" % camera_depth.type_id)
        # c_depth = carla.ColorConverter.LogarithmicDepth
        # camera_depth.listen(
        #     lambda image: image.save_to_disk("depth/%06d.png" % image.frame, c_depth)
        # )

        # DVS Camera
        # camera_dvs_bp = blueprint_library.find("sensor.camera.dvs")
        # camera_dvs_bp.set_attribute("sensor_tick", tick)
        # camera_dvs_transform = carla.Transform(
        #     carla.Location(x=1.5, z=2.4)
        # )  # without calibration
        # camera_dvs = world.spawn_actor(
        #     camera_dvs_bp, camera_dvs_transform, attach_to=ego_vehicle
        # )
        # sensor_list.append(camera_dvs)
        # print("created ego depth cam %s" % camera_dvs.type_id)
        # camera_dvs.listen(
        #     lambda image: Image.fromarray(dvs_cvt(image)).save(
        #         "./dvs/%06d.png" % image.frame
        #     )
        # )

        # Collision
        # collision_bp = blueprint_library.find("sensor.other.collision")
        # collision_transform = carla.Transform(
        #     carla.Location(x=ego_vehicle.bounding_box.extent.x, y=0, z=0)
        # )
        # collision = world.spawn_actor(
        #     collision_bp, collision_transform, attach_to=ego_vehicle
        # )
        # sensor_list.append(collision)
        # print("created ego collision %s" % collision.type_id)
        # colhis = []
        # collision.listen(lambda event: colhis.append(col(event)))
        # print("colhis")
        # print(colhis)

        time.sleep(5.0)

        ## loop
        while True:
            world_snapshot = world.wait_for_tick()

    finally:
        print("destroying actors")
        client.stop_recorder()

        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in sensor_list])
        print("done.")


if __name__ == "__main__":
    main()
