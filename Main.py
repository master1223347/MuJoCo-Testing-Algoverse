import mujoco
import mujoco.viewer

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("slope_box.xml")
data = mujoco.MjData(model)
start_pos = data.qpos[:].copy()  # Store initial state

with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(80000): 
        mujoco.mj_step(model, data)
        viewer.sync()
