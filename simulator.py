import os, time, math
import pybullet as p
import pybullet_data as pd

DT = 1.0/240.0

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)
p.loadURDF("plane.urdf")

p.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=65,
    cameraPitch=-55,
    cameraTargetPosition=[0.0, 0.0, 0.30]
)

# -------------------------
# Slightly transparent base (visible but subtle)
# -------------------------
base_pos = [0.0, 0.0, 0.55]  # higher up
base_half = [0.05, 0.05, 0.01]
base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half)
base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half, rgbaColor=[0.8, 0.8, 0.8, 0.25])  # alpha=0.25
base_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=base_col, baseVisualShapeIndex=base_vis,
                            basePosition=base_pos)

# If you want the base to be "ghosty" (no collisions), uncomment:
# p.setCollisionFilterGroupMask(base_id, -1, 0, 0)

# -------------------------
# Finger URDF: 2 links pointing down, bending about local Y (curl in local X-Z plane)
# -------------------------
L1, L2 = 0.06, 0.05
W1, H1 = 0.014, 0.014
W2, H2 = 0.013, 0.013

finger_urdf = f"""<?xml version="1.0"?>
<robot name="finger2r_down">
  <link name="base"/>

  <link name="link1">
    <inertial>
      <origin xyz="0 0 {-L1/2}"/>
      <mass value="0.15"/>
      <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {-L1/2}"/>
      <geometry><box size="{W1} {H1} {L1}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 {-L1/2}"/>
      <geometry><box size="{W1} {H1} {L1}"/></geometry>
    </collision>
  </link>

  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="25" velocity="8.0"/>
    <dynamics damping="0.05" friction="0.0"/>
  </joint>

  <link name="link2">
    <inertial>
      <origin xyz="0 0 {-L2/2}"/>
      <mass value="0.12"/>
      <inertia ixx="8e-5" iyy="8e-5" izz="8e-5" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {-L2/2}"/>
      <geometry><box size="{W2} {H2} {L2}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 {-L2/2}"/>
      <geometry><box size="{W2} {H2} {L2}"/></geometry>
    </collision>
  </link>

  <joint name="j2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 {-L1}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="25" velocity="8.0"/>
    <dynamics damping="0.05" friction="0.0"/>
  </joint>
</robot>
"""

tmp_path = os.path.join(os.getcwd(), "finger2r_down_tmp.urdf")
with open(tmp_path, "w") as f:
    f.write(finger_urdf)

# -------------------------
# Spawn 4 fingers in 2x2 square, yawed so they face the center
# + IMPORTANT CHANGES:
#   1) yaw += pi (flip 180 deg) so positive joint motion curls inward (if needed)
#   2) disable collisions between different fingers (so they don't fight each other)
# -------------------------
sx, sy = 0.08, 0.08  # spacing

square_offsets = [
    (-sx/2, -sy/2),
    (-sx/2, +sy/2),
    (+sx/2, -sy/2),
    (+sx/2, +sy/2),
]

colors = [
    [0.9, 0.2, 0.2, 1.0],
    [0.2, 0.9, 0.2, 1.0],
    [0.2, 0.2, 0.9, 1.0],
    [0.9, 0.9, 0.2, 1.0],
]

finger_ids = []
for k, (dx, dy) in enumerate(square_offsets):
    pos = [base_pos[0] + dx, base_pos[1] + dy, base_pos[2] - base_half[2] - 0.002]

    # yaw so local +X points toward center (0,0)
    # flip by 180 deg so positive joint angles curl toward the center
    yaw = math.atan2(-dy, -dx) + math.pi
    orn = p.getQuaternionFromEuler([0, 0, yaw])

    fid = p.loadURDF(tmp_path, basePosition=pos, baseOrientation=orn, useFixedBase=True)
    finger_ids.append(fid)

    for link_idx in range(p.getNumJoints(fid)):
        p.changeVisualShape(fid, link_idx, rgbaColor=colors[k])

    # disable default motors
    for j in range(p.getNumJoints(fid)):
        p.setJointMotorControl2(fid, j, p.VELOCITY_CONTROL, force=0)

# ---- Disable collisions between different fingers (all links) ----
for i in range(len(finger_ids)):
    for j in range(i+1, len(finger_ids)):
        A = finger_ids[i]
        B = finger_ids[j]
        linksA = [-1] + list(range(p.getNumJoints(A)))
        linksB = [-1] + list(range(p.getNumJoints(B)))
        for la in linksA:
            for lb in linksB:
                p.setCollisionFilterPair(A, B, la, lb, enableCollision=0)

# -------------------------
# Automatic open/close cycle (all fingers close toward center)
# -------------------------
open_j1, open_j2 = 0.0, 0.0
close_j1, close_j2 = 1.2, 1.25  # curl amount

cycle_hz = 0.25
w = 2.0 * math.pi * cycle_hz

t0 = time.time()
print("4 fingers facing inward, cycling open/close. Ctrl+C quits.")

try:
    while p.isConnected():
        t = time.time() - t0

        # smooth 0..1..0
        s = 0.5 * (1.0 - math.cos(w * t))

        q1 = (1 - s) * open_j1 + s * close_j1
        q2 = (1 - s) * open_j2 + s * close_j2

        for fid in finger_ids:
            p.setJointMotorControl2(fid, 0, p.POSITION_CONTROL, targetPosition=q1, force=35)
            p.setJointMotorControl2(fid, 1, p.POSITION_CONTROL, targetPosition=q2, force=35)

        p.stepSimulation()
        time.sleep(DT)

except KeyboardInterrupt:
    pass
