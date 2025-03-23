import mujoco
import mujoco.viewer
import time
import numpy as np

class Scene:
    def __init__():
        n: str
        n = input()
        scene = mujoco.from_xml_path("scene" + n + ".xml")
        scene_permissions = mujoco.from_json_path("scene" + n + ".xml")
        
         