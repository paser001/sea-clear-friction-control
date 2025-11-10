import pybullet as p
import pybullet_data
import time
import math

# --- Connect & setup ---
cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0/240.0)  # default is fine

# Visualizer tweaks
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0])

# --- World ---
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 0.5]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("cube_small.urdf", cubeStartPos, cubeStartOrientation)

# Reduce default damping so sliding is clearer
p.changeDynamics(boxId, -1, linearDamping=0.0, angularDamping=0.0)

# --- Sliders (live tuning) ---
mu_cube_slider = p.addUserDebugParameter("mu_lateral_cube", 0.0, 2.0, 0.3)
mu_plane_slider = p.addUserDebugParameter("mu_lateral_plane", 0.0, 2.0, 0.3)
mu_roll_slider  = p.addUserDebugParameter("mu_rolling", 0.0, 0.1, 0.01)
mu_spin_slider  = p.addUserDebugParameter("mu_spinning", 0.0, 0.1, 0.01)
tilt_slider     = p.addUserDebugParameter("tilt(rad)", 0.0, 0.3, 0.10)  # up to ~17 deg
nudge_slider    = p.addUserDebugParameter("nudge_force(N)", 0.0, 5.0, 0.5)

# Apply initial dynamics on both bodies
def apply_dynamics():
    mu_cube = p.readUserDebugParameter(mu_cube_slider)
    mu_plane = p.readUserDebugParameter(mu_plane_slider)
    mu_roll = p.readUserDebugParameter(mu_roll_slider)
    mu_spin = p.readUserDebugParameter(mu_spin_slider)

    # Apply on cube
    p.changeDynamics(boxId, -1,
                     lateralFriction=mu_cube,
                     rollingFriction=mu_roll,
                     spinningFriction=mu_spin,
                     linearDamping=0.0,
                     angularDamping=0.0)

    # Apply on plane
    p.changeDynamics(planeId, -1,
                     lateralFriction=mu_plane,
                     rollingFriction=mu_roll,
                     spinningFriction=mu_spin)

apply_dynamics()
last_vals = None


def setup_scene():
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    cubeStartPos = [0, 0, 0.5]
    cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF("cube_small.urdf", cubeStartPos, cubeStartOrientation)
    return planeId, boxId

planeId, boxId = setup_scene()


# Main loop
while p.isConnected():
    # Read slider values
    mu_cube = p.readUserDebugParameter(mu_cube_slider)
    mu_plane = p.readUserDebugParameter(mu_plane_slider)
    mu_roll = p.readUserDebugParameter(mu_roll_slider)
    mu_spin = p.readUserDebugParameter(mu_spin_slider)
    tilt = p.readUserDebugParameter(tilt_slider)
    nudge = p.readUserDebugParameter(nudge_slider)

    # Re-apply dynamics only when something changes
    current_vals = (mu_cube, mu_plane, mu_roll, mu_spin)
    if current_vals != last_vals:
        apply_dynamics()
        last_vals = current_vals

    # Tilt the plane
    p.resetBasePositionAndOrientation(
        planeId, [0, 0, 0],
        p.getQuaternionFromEuler([tilt, 0, 0])
    )

    # Gentle lateral push along +Y to observe motion under different frictions
    # (small continuous force; increase 'nudge_force' slider to see more motion)
    p.applyExternalForce(objectUniqueId=boxId, linkIndex=-1,
                         forceObj=[0, nudge, 0], posObj=[0,0,0], flags=p.WORLD_FRAME)

    p.stepSimulation()
    time.sleep(1.0/240.0)

    cubePos, _ = p.getBasePositionAndOrientation(boxId)
    print(f"Cube pos: {cubePos}")
    if cubePos[2] < 0.1:
        print("Cube fell off, resetting scene...")
        planeId, boxId = setup_scene()

# Optional (normally reached when GUI closed)
p.disconnect()
