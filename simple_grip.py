import pybullet as p
import pybullet_data
import time
import math

# --- Connect & setup ---
cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0/240.0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

# --- World ---
plane_id = p.loadURDF("plane.urdf")

# Cube to pick
cube_pos = [0.0, 0.0, 0.025]  # 5 cm cube half-height = 0.025
cube_id = p.loadURDF("cube_small.urdf", cube_pos, p.getQuaternionFromEuler([0,0,0]))
# Increase friction so the pinch holds better
p.changeDynamics(cube_id, -1, lateralFriction=2, rollingFriction=0.002, spinningFriction=0.002)

# dimensions (half extents)
PALM = [0.06, 0.01, 0.01]     # 12 x 2 x 2 cm
FINGER = [0.01, 0.01, 0.06]   # 2 x 2 x 12 cm (long along z)
CLEAR = 0.001  # tiny clearance so it doesnt touch


# collision and visuals
palm_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=PALM)
palm_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=PALM, rgbaColor=[0.7, 0.7, 0.7, 1])

finger_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=FINGER)
finger_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=FINGER, rgbaColor=[0.2, 0.2, 0.9, 1])

# Kinematic trick: mass=0 makes the base static/kinematic; we'll "animate" it via resetBasePosition...
palm_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=palm_col, baseVisualShapeIndex=palm_vis,
                            basePosition=[0.0, 0.0, 0.20])

left_id  = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=finger_col, baseVisualShapeIndex=finger_vis,
                             basePosition=[-0.05, 0.0, 0.15])
right_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=finger_col, baseVisualShapeIndex=finger_vis,
                             basePosition=[+0.05, 0.0, 0.15])

# Friction on fingers to improve grip
for bid in (left_id, right_id):
    p.changeDynamics(bid, -1, lateralFriction=2.0, rollingFriction=0.002, spinningFriction=0.002)

# Camera
p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=40, cameraPitch=-35,
                             cameraTargetPosition=[0.0, 0.0, 0.05])

# Helper to place gripper pieces given palm pose and opening width
def set_gripper(palm_xyz, opening_width):
    px, py, pz = palm_xyz
    # desired half-gap between inner finger faces
    half_gap = max(0.0, opening_width / 2.0)

    # Fingers are boxes with half-width FINGER[0].
    # To achieve an inner-face gap = opening_width,
    # place finger centers at: ±(half_gap + FINGER[0] + CLEAR) from center.
    left_x  = px - (half_gap + FINGER[0] + CLEAR)
    right_x = px + (half_gap + FINGER[0] + CLEAR)

    # Keep the fingers vertically just below the palm
    finger_z = pz - (PALM[2] + FINGER[2]) + 0.002

    # Set poses
    p.resetBasePositionAndOrientation(palm_id,  [px, py, pz],  [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(left_id,  [left_x,  py, finger_z], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(right_id, [right_x, py, finger_z], [0, 0, 0, 1])


# Simple motion primitives
def descend(z_from, z_to, opening, steps=240):
    for i in range(steps):
        z = z_from + (z_to - z_from) * (i+1)/steps
        set_gripper([0.0, 0.0, z], opening)
        p.stepSimulation(); time.sleep(1/240)

def ascend(z_from, z_to, opening, steps=360):
    for i in range(steps):
        z = z_from + (z_to - z_from) * (i+1)/steps
        set_gripper([0.0, 0.0, z], opening)
        p.stepSimulation(); time.sleep(1/240)

def close_width(from_w, to_w, at_z, steps=240):
    for i in range(steps):
        w = from_w + (to_w - from_w) * (i+1)/steps
        set_gripper([0.0, 0.0, at_z], w)
        p.stepSimulation(); time.sleep(1/240)

# --- Scripted pick sequence ---
open_w = 0.08   # 8 cm fully open
close_w = 0.043 # ~3.5 cm to pinch a small cube
hover_z = 0.20
grasp_z = 0.08  # fingers around the cube sides (cube top is at z ~ 0.05)

set_gripper([0.0, 0.0, hover_z], open_w)
for _ in range(240): p.stepSimulation(); time.sleep(1/240)

descend(hover_z, grasp_z, open_w, steps=480)
close_width(open_w, close_w, grasp_z, steps=360)
ascend(grasp_z, 0.28, close_w, steps=600)

# Keep running; press 'r' to reset pose; 'o' to open; 'c' to close a bit more; d descend and a ascend
while p.isConnected():
    keys = p.getKeyboardEvents()
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        set_gripper([0.0, 0.0, hover_z], open_w)
    if ord('o') in keys and keys[ord('o')] & p.KEY_WAS_TRIGGERED:
        open_w = 0.08
        set_gripper([0.0, 0.0, hover_z], open_w)
    if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
        close_w = max(0.02, close_w - 0.005)  # a bit tighter
        set_gripper([0.0, 0.0, hover_z], close_w)
    if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
        descend(hover_z, grasp_z, open_w, steps=480)
    if ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED:
        ascend(grasp_z, 0.28, close_w, steps=600)
    p.stepSimulation()
    time.sleep(1/240)
