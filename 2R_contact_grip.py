
import os, time, math
import pybullet as p
import pybullet_data as pd
import numpy as np

# -------------------------
# Config
# -------------------------
L1, L2 = 0.15, 0.15          # link lengths (m)
FN_TARGET = 5.0              # desired normal force on wall (N)
FT_TANGENTIAL = 1.0          # desired tangential push while sliding (N, along +X)
KP_F, KI_F = 2.0, 5.0        # PI gains for normal-force control
DT = 1.0/240.0

# -------------------------
# Start PyBullet
# -------------------------
cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)

# Make contacts "stickier" and stable
p.setPhysicsEngineParameter(numSolverIterations=200, numSubSteps=2, contactERP=0.3, frictionERP=0.3)

# Ground (optional visual reference)
p.loadURDF("plane.urdf")

# -------------------------
# Create a vertical wall at y = 0
# -------------------------
wall_hx, wall_hy, wall_hz = 0.5, 0.005, 0.4
wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_hx, wall_hy, wall_hz])
wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_hx, wall_hy, wall_hz], rgbaColor=[0.8,0.8,0.8,1])
wall_id  = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_col, baseVisualShapeIndex=wall_vis,
                             basePosition=[0.0, 0.0, wall_hz])
p.changeDynamics(wall_id, -1, lateralFriction=1.5, restitution=0.0, frictionAnchor=1)

# Camera
# Move wall a bit back so it's not exactly on y=0
p.resetBasePositionAndOrientation(wall_id, [0.0, -0.02, wall_hz], [0,0,0,1])

# Make wall semi-transparent
p.changeVisualShape(wall_id, -1, rgbaColor=[0.8, 0.8, 0.8, 0.3])

# Place camera high and to the side, looking at the fingertip area
p.resetDebugVisualizerCamera(
    cameraDistance=1.4,
    cameraYaw=80,          # rotate around Z (try 60–120)
    cameraPitch=-25,       # negative looks down
    cameraTargetPosition=[0.15, 0.05, 0.05]
)

def set_camera_top():
    p.resetDebugVisualizerCamera(cameraDistance=1.0,
                                 cameraYaw=90,
                                 cameraPitch=-89,
                                 cameraTargetPosition=[0.1, 0.05, 0.05])

def set_camera_side():
    p.resetDebugVisualizerCamera(cameraDistance=1.2,
                                 cameraYaw=60,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[0.1, 0.05, 0.05])
    

def sleep_temp(t):
    for _ in range(t):
        p.stepSimulation()
        time.sleep(1./240.)


# Optional: enable wireframe to see through objects
# p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)


# -------------------------
# Write a tiny URDF for a 2R planar arm (revolute about Z)
# Link frames: each link's visual/collision is centered; joint offsets place them tip-to-tip.
# -------------------------
urdf = f"""<?xml version="1.0"?>
<robot name="finger2r">
  <link name="base"/>
  <link name="link1">
    <inertial><origin xyz="{L1/2} 0 0"/><mass value="0.2"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4"/></inertial>
    <visual><origin xyz="{L1/2} 0 0"/><geometry><box size="{L1} 0.02 0.02"/></geometry></visual>
    <collision><origin xyz="{L1/2} 0 0"/><geometry><box size="{L1} 0.02 0.02"/></geometry></collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="6.0"/>
  </joint>

  <link name="link2">
    <inertial><origin xyz="{L2/2} 0 0"/><mass value="0.2"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4"/></inertial>
    <visual><origin xyz="{L2/2} 0 0"/><geometry><box size="{L2} 0.018 0.018"/></geometry></visual>
    <collision><origin xyz="{L2/2} 0 0"/><geometry><box size="{L2} 0.018 0.018"/></geometry></collision>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="{L1} 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="6.0"/>
  </joint>
</robot>
"""

# Creating tip
tip_radius = 0.005  # 5 mm
tip_col = p.createCollisionShape(p.GEOM_SPHERE, radius=tip_radius)
tip_vis = p.createVisualShape(p.GEOM_SPHERE, radius=tip_radius, rgbaColor=[1,0,0,1])
tip_id  = p.createMultiBody(baseMass=0.01,      # small but non-zero mass
                            baseCollisionShapeIndex=tip_col,
                            baseVisualShapeIndex=tip_vis,
                            basePosition=[0,0,0])
# Grip
p.changeDynamics(tip_id, -1, lateralFriction=2.5, restitution=0.0, frictionAnchor=1)




tmp_path = os.path.join(os.getcwd(), "finger_2r.urdf")
with open(tmp_path, "w") as f:
    f.write(urdf)

arm_id = p.loadURDF(tmp_path, basePosition=[0.15, 0.20, 0.05], useFixedBase=True)
# joint indices: 0 -> joint1, 1 -> joint2
EE_LINK = 1  # link2
LOCAL_TIP = [L2/2, 0, 0]  # fingertip position in link2 frame (relative to COM)

# Friction on the "finger tip" (link2) to interact with wall
p.changeDynamics(arm_id, EE_LINK, lateralFriction=2.5, restitution=0.0, frictionAnchor=1)

cid = p.createConstraint(parentBodyUniqueId=arm_id, parentLinkIndex=EE_LINK,
                         childBodyUniqueId=tip_id,  childLinkIndex=-1,
                         jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
                         parentFramePosition=LOCAL_TIP, childFramePosition=[0,0,0])


# Disable default motors; we'll command torques
for j in (0,1):
    p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)

# Move roughly in front of wall (position control just to get there)
def goto(q1, q2, steps=480):
    for j, q in enumerate([q1,q2]):
        p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, targetPosition=q, force=20, positionGain=0.1, velocityGain=1.0)
    for _ in range(steps):
        p.stepSimulation(); time.sleep(DT)
    # release motors for torque control
    for j in (0,1):
        p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)

# Initial pose: reach to near wall
goto(q1=-0.59695293, q2=-0.39810182, steps=1)
sleep_temp(240)



# Helper: end-effector world pose
def ee_pose():
    pos, orn, _, _, _, _ = p.getLinkState(arm_id, EE_LINK, computeForwardKinematics=True)
    return pos, orn

# Compute Jacobian at a point on the fingertip (end of link2)
# link frame has COM at L2/2 along +X; fingertip is at local +X by another L2/2
LOCAL_TIP = [L2/2, 0, 0]  # relative to link2 COM

def get_jacobian(q, qd):
    Jlin, Jang = p.calculateJacobian(arm_id, EE_LINK, LOCAL_TIP, list(q), list(qd), [0.0, 0.0])
    J = np.array(Jlin)  # 3 x n
    return J  # linear Jacobian

# Measure normal contact force (sum) on end-effector with the wall (normal ~ +Y on wall, so force on EE is -Y)
def measure_normal_force():
    cps = p.getContactPoints(bodyA=arm_id, linkIndexA=EE_LINK, bodyB=wall_id)
    # Bullet reports "normal force" magnitude; normal is from B->A (wall to EE).
    fn = 0.0
    for cp in cps:
        fn += cp[9]  # normalForce
    return fn

# -------------------------
# Control loop
# - Push into wall to achieve FN_TARGET (normal along -Y),
# - While holding FN, apply tangential force along +X to slide.
# -------------------------
err_int = 0.0
slide = True
i = 0
print("Running. Press Ctrl+C to stop.")

try:
    while p.isConnected():
        keys = p.getKeyboardEvents()
        if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
            set_camera_top()

        if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
            set_camera_side()
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            goto(q1=-0.59695293, q2=-0.39810182, steps=240)
            err_int = 0.0
            i = 0
            sleep_temp(240)
            print("Reset done")

        

        # Read state
        js0 = p.getJointState(arm_id, 0)
        js1 = p.getJointState(arm_id, 1)
        q = np.array([js0[0], js1[0]])
        qd = np.array([js0[1], js1[1]])

        cps = p.getContactPoints(bodyA=arm_id, linkIndexA=EE_LINK, bodyB=wall_id)
        #print("num contacts:", len(cps))
        if len(cps)<=2:
            print("contacts", len(cps), "angles:", q)
            if len(cps)==1:
                print("contact Y positions:", [cp[6][1] for cp in cps])  # contact point on fingertip

        # # Jacobian at fingertip
        J = get_jacobian(q, qd)  # 3x2


        fn_meas = measure_normal_force()
        e = FN_TARGET - fn_meas
        err_int += e * DT
        fn_cmd = KP_F * e + KI_F * err_int
        fn_cmd = max(0.0, min(fn_cmd, 20.0)) 



        if fn_meas > 0.5:  # touching the wall
            p.changeVisualShape(arm_id, 1, rgbaColor=[1, 0, 0, 1])   # red 
        else:
            p.changeVisualShape(arm_id, 1, rgbaColor=[0.2, 0.8, 0.2, 1])  # green 


        # [-0.59695293 -0.39810182]  
        # -45 , -15 

        goto(q1=-0.59695293, q2=-0.39810182 - math.radians(i), steps=1)


        # # Tangential force (slide along +X) if in contact
        # ft_cmd = FT_TANGENTIAL if slide and fn_meas > 0.5 else 0.0

        # # Desired EE force vector f = [Fx, Fy, Fz]
        # # Wall normal points +Y from wall to EE; we push EE toward -Y, so Fy = -fn_cmd
        # f_des = np.array([ft_cmd, -fn_cmd, 0.0])  # (N)

        # # Map to joint torques: tau = J^T * f
        # tau = J.T @ f_des

        # # Apply torques
        # p.setJointMotorControlArray(arm_id, [0,1], p.TORQUE_CONTROL, forces=tau.tolist())
        i= i + 0.1
        p.stepSimulation()
        time.sleep(DT)

except KeyboardInterrupt:
    pass