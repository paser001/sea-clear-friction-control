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

# Small cube to grasp
cube_start_pos = [0.6, 0.0, 0.02]   # in front of robot
cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, p.getQuaternionFromEuler([0,0,0]), globalScaling=1.0)

# --- Panda robot with gripper ---
# Note: This URDF is included in pybullet_data
panda_urdf = "franka_panda/panda.urdf"
panda_id = p.loadURDF(panda_urdf, useFixedBase=True)

# Get joint info to find indices by name (robust to URDF changes)
name_to_index = {}
for j in range(p.getNumJoints(panda_id)):
    info = p.getJointInfo(panda_id, j)
    name_to_index[info[1].decode("utf-8")] = j

# Common indices for Panda (but we still looked them up above)
arm_joint_names = [
    "panda_joint1","panda_joint2","panda_joint3","panda_joint4",
    "panda_joint5","panda_joint6","panda_joint7"
]
finger_left = name_to_index.get("panda_finger_joint1")
finger_right = name_to_index.get("panda_finger_joint2")

# End effector link index (usually 11 for Panda)
ee_name = "panda_hand"  # link name
ee_index = name_to_index.get(ee_name, 11)

# Disable default motors and use our own control
for j in range(p.getNumJoints(panda_id)):
    p.setJointMotorControl2(panda_id, j, p.VELOCITY_CONTROL, force=0)

# Helper: set arm (7 joints) to target positions
def set_arm(q):
    for i, jname in enumerate(arm_joint_names):
        p.setJointMotorControl2(
            panda_id,
            name_to_index[jname],
            p.POSITION_CONTROL,
            targetPosition=q[i],
            force=200,
            positionGain=0.04,
            velocityGain=1.0
        )

# Helper: open/close gripper (symmetric)
# Opening width is ~0.08 m max (0.04 per finger). Use 0.04 for fully open, 0.0 closed.
def set_gripper(width):
    half = max(0.0, min(0.04, width/2.0))
    p.setJointMotorControl2(panda_id, finger_left,  p.POSITION_CONTROL, targetPosition=half, force=20)
    p.setJointMotorControl2(panda_id, finger_right, p.POSITION_CONTROL, targetPosition=half, force=20)

# IK move (position + optional orientation)
def move_ee(target_pos, target_rpy=None):
    if target_rpy is None:
        target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])  # wrist facing down
    else:
        target_orn = p.getQuaternionFromEuler(target_rpy)
    q = p.calculateInverseKinematics(
        panda_id, ee_index, target_pos, target_orn,
        lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        upperLimits=[ 2.8973,  1.7628,  2.8973,  3.0718,  2.8973,  3.7525,  2.8973],
        jointRanges=[ 5.8,     3.5,     5.8,     6.1,     5.8,     3.77,    5.8   ],
        restPoses =[ 0,       -0.3,     0,      -2.0,     0,       2.0,     0     ],
        maxNumIterations=200,
        residualThreshold=1e-4
    )
    set_arm(q[:7])

# Camera
p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=60, cameraPitch=-35, cameraTargetPosition=[0.6, 0.0, 0.0])

# --- Sequence: open, approach, grasp, lift ---

# 1) Open gripper
set_gripper(0.08)
for _ in range(240): p.stepSimulation(); time.sleep(1/240)

# 2) Move above cube
hover_pos = [cube_start_pos[0], cube_start_pos[1], cube_start_pos[2] + 0.20]
move_ee(hover_pos)
for _ in range(480): p.stepSimulation(); time.sleep(1/240)

# 3) Move down to grasp height
grasp_pos = [cube_start_pos[0], cube_start_pos[1], cube_start_pos[2] + 0.02]
move_ee(grasp_pos)
for _ in range(480): p.stepSimulation(); time.sleep(1/240)

# 4) Close gripper to grasp
set_gripper(0.0)  # close
for _ in range(720): p.stepSimulation(); time.sleep(1/240)

# 5) Lift up
lift_pos = [cube_start_pos[0], cube_start_pos[1], cube_start_pos[2] + 0.30]
move_ee(lift_pos)
for _ in range(960): p.stepSimulation(); time.sleep(1/240)

# Keep sim running (press Ctrl+C in terminal or close window)
while p.isConnected():
    p.stepSimulation()
    time.sleep(1/240)
