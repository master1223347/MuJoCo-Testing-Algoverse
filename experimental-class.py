import mujoco
import mujoco.viewer
import numpy as np

class ExperimentalMuJoCoScene:
    def __init__(self, model_path, sim_time=5.0, render=True):
        """
        Initialize the MuJoCo environment for an experimental scene.

        Parameters:
        - model_path: str, path to the XML model file defining the MuJoCo scene
        - sim_time: float, duration of the simulation (in seconds)
        - render: bool, whether to render the scene during the simulation
        """
        self.model = mujoco.MjModel.load_model_from_path(model_path)
        self.sim = mujoco.MjSim(self.model)
        self.viewer = None
        if render:
            self.viewer = mujoco.MjViewer(self.sim)
        self.sim_time = sim_time
        self.time_step = self.sim.model.opt.timestep  # Time step of the simulation

    def reset(self):
        """Reset the simulation to its initial state."""
        self.sim.reset()

    def step(self):
        """Advance the simulation by one time step."""
        self.sim.step()

    def render(self):
        """Render the current state of the scene."""
        if self.viewer:
            self.viewer.render()

    def run(self):
        """Run the simulation for the given time."""
        total_steps = int(self.sim_time / self.time_step)
        for _ in range(total_steps):
            self.step()
            self.render()

    def get_state(self):
        """Return the current state of the simulation."""
        return self.sim.get_state()

    def apply_action(self, action):
        """Apply an action to the scene (e.g., control joints or tendons)."""
        # Action could be a numpy array corresponding to the control inputs
        # For example, if controlling robot joints:
        self.sim.data.ctrl[:] = action

    def close(self):
        """Close the viewer if it's being used."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    try:
        scene.run()
    finally:
        scene.close()
