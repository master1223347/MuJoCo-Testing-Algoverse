#!/usr/bin/env python3
"""
Simulator Module for MuJoCo-based Experiments

This module defines the Simulator class, which encapsulates loading,
stepping, rendering, querying, and managing MuJoCo simulation scenes.

Key Features:
  - Scene loading via XML identifiers
  - Passive viewer rendering
  - Simulation control (step, reset, scene_step, close)
  - Body kinematics: positions, velocities, accelerations
  - Dynamics queries: displacement, force, torque,
    kinetic/potential energy, momentum, angular momentum
  - Collision detection utilities with optional elastic response
  - Permissions-based parameter access
  - Center-of-Mass and inertia calculations
  - Trajectory recording and export
  - Quaternion ↔ rotation matrix & Euler conversions
  - Relative and absolute object movement
  - Context manager support

Usage Example:
    from simulator import Simulator

    with Simulator("Scene_1") as sim:
        sim.step(0.1)
        pos, t = sim.get_position("pendulum")
        sim.apply_force("pendulum", [0,0,-9.81])
        disp = sim.compute_displacement("pendulum")
        params = sim.get_parameters("pendulum")
        sim.record_trajectory("pendulum", duration=2.0, dt=0.01)
        sim.export_trajectory("pendulum", "trajectory.csv")
        frame = sim.render()

Dependencies:
    mujoco, numpy, csv

"""
from __future__ import annotations
import csv
from pathlib import Path
import logging
import numpy as np
import mujoco
import mujoco.viewer
from typing import Any, Dict, List, Tuple, Union

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | [%(levelname)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
)
if not logger.hasHandlers():
    logger.addHandler(handler)

class Simulator:
    """
    Simulator encapsulates a MuJoCo environment for a given scene.

    Attributes:
        scene_id: Scene identifier.
        scenes_dir: Base directory for scene XMLs.
        model, data: MuJoCo model and data.
        viewer: Passive viewer instance.
        time: Elapsed simulation time.
        initial_qpos: Snapshot of initial positions.
        trajectories: Recorded trajectories per object.
        permissions: Optional access control dict.
    """

    def __init__(
        self,
        scene_id: str,
        scenes_dir: Union[str, Path] = None
    ) -> None:
        self.scene_id = scene_id
        base = scenes_dir or Path(__file__).parent / "Scenes"
        self.scenes_dir = Path(base)
        self.model_path = self._resolve_model_path()
        self.trajectories: Dict[str, List[Tuple[float, np.ndarray]]] = {}
        self.permissions: Dict[Union[int,str], Dict[str,bool]] = {}

        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.data = mujoco.MjData(self.model)
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.time = 0.0
            self.initial_qpos = np.copy(self.data.qpos)
            logger.info("Loaded scene '%s' from %s", scene_id, self.model_path)
        except Exception as e:
            logger.error("Initialization error for scene '%s': %s", scene_id, e)
            raise

    def _resolve_model_path(self) -> Path:
        xml_file = self.scenes_dir / self.scene_id / f"{self.scene_id}.xml"
        if not xml_file.is_file():
            msg = f"Scene XML missing: {xml_file}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        return xml_file

    def reset(self) -> None:
        """Reset state: positions, velocities, time, trajectories."""
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.time = 0.0
        self.trajectories.clear()
        logger.info("Simulation reset to time 0.")

    # Alias for original name
    reset_sim = reset

    def step(self, duration: float) -> None:
        """Advance simulation by duration, syncing viewer."""
        dt = self.model.opt.timestep
        steps = int(np.floor(duration / dt))
        rem = duration - steps * dt
        for _ in range(steps + (1 if rem > 1e-9 else 0)):
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()
        self.time += duration
        logger.debug("Stepped %.4f sec (total %.4f)", duration, self.time)

    def scene_step(self) -> str:
        """Step exactly one timestep."""
        dt = self.model.opt.timestep
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        self.time += dt
        return f"Scene stepped by {dt:.6f} seconds."

    def render(self) -> np.ndarray:
        """Capture current frame."""
        img = self.viewer.capture_frame()
        logger.debug("Frame captured at t=%.4f", self.time)
        return img

    def _body_index(self, object_id: Union[int,str]) -> int:
        if isinstance(object_id, int):
            return object_id
        try:
            return self.model.body_name2id(object_id)
        except KeyError:
            raise ValueError(f"Unknown body '{object_id}'")

    def get_position(self, object_id: Union[int,str]) -> Tuple[np.ndarray,float]:
        idx = self._body_index(object_id)
        return self.data.xpos[idx].copy(), self.time

    def get_velocity(self, object_id: Union[int,str]) -> np.ndarray:
        idx = self._body_index(object_id)
        return self.data.qvel[idx*6:idx*6+3].copy()

    def get_acceleration(self, object_id: Union[int,str]) -> np.ndarray:
        v0 = self.get_velocity(object_id)
        mujoco.mj_step(self.model, self.data)
        v1 = self.get_velocity(object_id)
        dt = self.model.opt.timestep
        return (v1 - v0) / dt

    def compute_displacement(self, object_id: Union[int,str]) -> float:
        """Euclidean displacement since reset."""
        idx = self._body_index(object_id)
        pos0 = self.initial_qpos[idx*7:idx*7+3]
        pos1 = self.data.xpos[idx].copy()
        return float(np.linalg.norm(pos1 - pos0))

    def apply_force(self, object_id: Union[int,str], force: List[float]) -> None:
        idx = self._body_index(object_id)
        self.data.xfrc_applied[idx,:3] = force
        logger.debug("Force %s applied to '%s'", force, object_id)

    def apply_torque(self, object_id: Union[int,str], torque: List[float]) -> None:
        idx = self._body_index(object_id)
        self.data.xfrc_applied[idx,3:] = torque
        logger.debug("Torque %s applied to '%s'", torque, object_id)

    def set_velocity(self, object_id: Union[int,str], vel: List[float]) -> None:
        idx = self._body_index(object_id)
        self.data.qvel[idx*6:idx*6+3] = vel
        mujoco.mj_forward(self.model, self.data)
        logger.info("Velocity of '%s' set to %s", object_id, vel)

    def set_position(self, object_id: Union[int,str], pos: List[float]) -> None:
        idx = self._body_index(object_id)
        self.data.qpos[idx*7:idx*7+3] = pos
        mujoco.mj_forward(self.model, self.data)
        logger.info("Position of '%s' set to %s", object_id, pos)

    # original move_object alias
    move_object = set_position

    def detect_collision(self, a: Union[int,str], b: Union[int,str]) -> bool:
        ia, ib = self._body_index(a), self._body_index(b)
        collided = False
        for c in self.data.contact:
            if (c.geom1, c.geom2) in ((ia,ib),(ib,ia)):
                logger.info("Collision detected between %s and %s", a, b)
                collided = True
                break
        return collided

    def compute_force(self, object_id: Union[int,str], mass: float) -> np.ndarray:
        acc = self.get_acceleration(object_id)
        return mass * acc

    def compute_momentum(self, object_id: Union[int,str], mass: float) -> np.ndarray:
        vel = self.get_velocity(object_id)
        return mass * vel

    def get_torque(self, object_id: Union[int,str]) -> Dict[str,float]:
        idx = self._body_index(object_id)
        t = self.data.qfrc_applied[idx*6+3:idx*6+6]
        return {k:v for k,v in zip(("x","y","z"), t.tolist())}

    def compute_kinetic_energy(self, object_id: Union[int,str], mass: float) -> float:
        v = self.get_velocity(object_id)
        return 0.5 * mass * float(np.dot(v,v))

    def compute_potential_energy(
        self, object_id: Union[int,str], mass: float,
        gravity: float = 9.81
    ) -> float:
        pos, _ = self.get_position(object_id)
        return mass * gravity * float(pos[2])

    def compute_angular_momentum(self, object_id: Union[int,str], mass: float) -> np.ndarray:
        pos,_ = self.get_position(object_id)
        p = self.compute_momentum(object_id, mass)
        return np.cross(pos, p)

    def set_permissions(self, permissions: Dict[Union[int,str], Dict[str,bool]]) -> None:
        """Define access rules for get_parameters."""
        self.permissions = permissions
        logger.info("Permissions updated.")

    def get_parameters(self, object_id: Union[int,str]) -> Dict[str,Any]:
        idx = self._body_index(object_id)
        perms = self.permissions.get(object_id, {})
        if not perms.get("get_parameters", True):
            raise PermissionError(f"Access denied for parameters of {object_id}")
        return {
            "mass": float(self.model.body_mass[idx]),
            "bounding_box": self.model.body_inertia[idx].tolist(),
            "type": int(self.model.body_parentid[idx])
        }

    def change_position(
        self,
        object_id: Union[int,str],
        dx: float,
        dy: float,
        dz: float,
        in_world_frame: bool = True
    ) -> Dict[str,Tuple[float,float,float]]:
        """Shift position by (dx,dy,dz) in world/local frame."""
        idx = self._body_index(object_id)
        px,py,pz = self.data.qpos[idx*7:idx*7+3]
        if in_world_frame:
            new = np.array([px+dx, py+dy, pz+dz])
        else:
            quat = self.data.qpos[idx*7+3:idx*7+7]
            R = self.quat_to_rot_matrix(quat)
            new = np.array([px,py,pz]) + R @ np.array([dx,dy,dz])
        self.data.qpos[idx*7:idx*7+3] = new
        mujoco.mj_forward(self.model, self.data)
        return {"new_position": tuple(new.tolist())}

    @staticmethod
    def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
        q = q/np.linalg.norm(q)
        w,x,y,z = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+ y*y)]
        ])

    @staticmethod
    def rot_matrix_to_euler(R: np.ndarray) -> Tuple[float,float,float]:
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy<1e-6
        if not singular:
            x = np.arctan2(R[2,1],R[2,2])
            y = np.arctan2(-R[2,0],sy)
            z = np.arctan2(R[1,0],R[0,0])
        else:
            x = np.arctan2(-R[1,2],R[1,1])
            y = np.arctan2(-R[2,0],sy)
            z = 0.0
        return x,y,z

    def get_center_of_mass(self) -> np.ndarray:
        masses = self.model.body_mass
        pos = self.data.xpos
        total = np.sum(masses)
        return np.sum(masses[:,None]*pos,axis=0)/total

    def record_trajectory(
        self,
        object_id: Union[int,str],
        duration: float,
        dt: float
    ) -> None:
        idx = self._body_index(object_id)
        traj: List[Tuple[float,np.ndarray]] = []
        steps = int(np.floor(duration/dt))
        for _ in range(steps):
            pos,t = self.get_position(object_id)
            traj.append((t,pos.copy()))
            self.step(dt)
        self.trajectories[object_id] = traj
        logger.info("Recorded %d points for '%s'",len(traj),object_id)

    def export_trajectory(
        self,
        object_id: Union[int,str],
        filepath: Union[str,Path]
    ) -> None:
        path = Path(filepath)
        traj = self.trajectories.get(object_id)
        if not traj:
            raise ValueError(f"No trajectory for '{object_id}'")
        with path.open('w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time','x','y','z'])
            for t,pos in traj:
                writer.writerow([f"{t:.5f}",*pos.tolist()])
        logger.info("Trajectory for '%s' saved to %s",object_id,path)

    def close(self) -> None:
        """Clean up viewer and resources."""
        if hasattr(self,'viewer') and self.viewer:
            self.viewer.close()
            logger.info("Viewer closed.")

    def __enter__(self) -> Simulator:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
