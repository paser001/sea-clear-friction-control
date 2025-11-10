import os, time, math
import pybullet as p
import pybullet_data as pd
import numpy as np 
import csv
from datetime import datetime
from scipy.signal import savgol_filter
from collections import deque

from scipy.signal import butter, sosfilt, sosfilt_zi

# -------------------------
# Config
# -------------------------
L1, L2 = 0.15, 0.15          # link lengths (m)
#Kp, Kd = 30, 2      
Kp, Kd = 5, 0.1
TMAX = 2.0
A = math.radians(30)    # amplitude (±30 deg)
w = 0.3                 # rad/s (slow to reduce inertia effects)
RUN_SECS = 50.0
DT = 1.0/240.0
DOF = 2                 
JIDX = 0                  # identify friction on joint 1
OTHER = 1
t0 = time.time()
log = []
WIN = 31         # window size (must be odd)
POLY = 3         # polynomial order (2 or 3 works well)

START_POS = [0 , 0]

q_hist  = deque(maxlen=WIN)
t_hist  = deque(maxlen=WIN)
# -------------------------
# Start PyBullet
# -------------------------
cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81) # WITH GRAVITY
# p.setGravity(0,0,0)      # NO GRAVITY
p.setTimeStep(DT)

# Make contacts frictionmaxxin and stable
p.setPhysicsEngineParameter(numSolverIterations=200, numSubSteps=2, contactERP=0.3, frictionERP=0.3)
p.loadURDF("plane.urdf")

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


# -------------------------
# URDF for the 2R finger
# -------------------------
urdf = f"""<?xml version="1.0"?>
<robot name="finger2r">
  <link name="base"/>
  <link name="link1">
    <inertial><origin xyz="{L1/2} 0 0"/><mass value="0.5"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4"/></inertial>
    <visual><origin xyz="{L1/2} 0 0"/><geometry><box size="{L1} 0.02 0.02"/></geometry></visual>
    <collision><origin xyz="{L1/2} 0 0"/><geometry><box size="{L1} 0.02 0.02"/></geometry></collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <dynamics damping="0.0" friction="0.0"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="6.0"/>
  </joint>

  <link name="link2">
    <inertial><origin xyz="{L2/2} 0 0"/><mass value="0.01"/><inertia ixx="1e-6" iyy="1e-6" izz="1e-6"/></inertial>
    <visual><origin xyz="{L2/2} 0 0"/><geometry><box size="{L2} 0.018 0.018"/></geometry></visual>
    <collision><origin xyz="{L2/2} 0 0"/><geometry><box size="{L2} 0.018 0.018"/></geometry></collision>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="{L1} 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <dynamics damping="0.0" friction="0.0"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="6.0"/>
  </joint>
</robot>
"""

tmp_path = os.path.join(os.getcwd(), "finger_2r.urdf")
with open(tmp_path, "w") as f:
    f.write(urdf)

arm_id = p.loadURDF(tmp_path, basePosition=[0.15, 0.20, 0.05], useFixedBase=True)

# REFERENCE ARM (ghost, no collisions)
ghost_id = p.loadURDF(tmp_path, basePosition=[0.15, 0.20, 0.05], useFixedBase=True)

for link_idx in [-1] + list(range(p.getNumJoints(ghost_id))):
    p.setCollisionFilterGroupMask(ghost_id, link_idx, 0, 0)

p.changeVisualShape(ghost_id, -1, rgbaColor=[0.2, 0.6, 1.0, 0.35])  # base
for link_idx in range(p.getNumJoints(ghost_id)):
    p.changeVisualShape(ghost_id, link_idx, rgbaColor=[0.2, 0.6, 1.0, 0.35])


# joint indices: 0 -> joint1, 1 -> joint2
EE_LINK = 1  # link2
LOCAL_TIP = [L2/2, 0, 0]  # fingertip position in link2 frame (relative to COM)

# # Friction on the "finger tip" (link2) to interact with wall
# p.changeDynamics(arm_id, EE_LINK, lateralFriction=2.5, restitution=0.0, frictionAnchor=1)


# Disable default motor to command torques
for j in (0,1):
    p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)

# Move to an angle positon over some steps
def goto(q1, q2, steps=480):
    for j, q in enumerate([q1,q2]):
        p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, targetPosition=q, force=20, positionGain=0.1, velocityGain=1.0)
    for _ in range(steps):
        p.stepSimulation(); time.sleep(DT)
    # release motors for torque control
    for j in (0,1):
        p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)

# ---------- Hidden "true" friction (plant) ----------
# Stribeck: tau_f(qd) = Fc*sign(qd) + Fv*qd + (Fs-Fc)*exp( -( |qd|/vs )^2 )
Fc_true = 0.1     # Nm  (Coulomb)
Fs_true = 0.12     # Nm  (static peak)
Fv_true = 0.02     # Nms/rad (viscous)
vs_true = 0.10     # rad/s  (Stribeck speed)
noise_sigma = 0.01 # Nm (torque noise)
Fc_simple = 0.15      # Coulomb friction [Nm]
Fv_simple = 0.02      # viscous friction [Nms/rad]
v_eps_simple = 0.02   # smoothing speed [rad/s]

def tau_f_simple(qd):
    return Fc_simple * np.tanh(qd / v_eps_simple) + Fv_simple * qd


def tau_f_true(qd):
    mag = Fc_true + (Fs_true - Fc_true) * np.exp(-(abs(qd)/vs_true)**2)
    return mag * np.sign(qd) + Fv_true * qd


# ---------- Velocity plateau generator ----------
# CHATGPT GENERATED IDK WHAT BRO MEANT WITH THIS
# Target speeds (rad/s) and per-plateau hold duration (s)
speeds = [ 0.05, 0.10, 0.20, 0.40, 0.80 ]
plateau_hold = 5.0   # seconds per +v and -v
tau_ramp = 0.5       # seconds, smooth approach to new target speed

def schedule_plateaus():  
    seq = []
    for v in speeds:
        seq.append(+v); seq.append(-v)
    return seq

# smooth first-order ramp: v_ref <- v_ref + alpha*(v_target - v_ref)
alpha = DT / tau_ramp

# ---------- Disable default motors, center pose ----------
for j in (0,1):
    p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)
# Initial pose: reach to near wall # WALL IS GONE :(
# goto(q1=-0.59695293, q2=-0.39810182, steps=1)
goto(q1=START_POS[0], q2=START_POS[1])
sleep_temp(480)
for j in (0,1): 
    p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)



# ---------- Logging setup ----------

os.makedirs("csvs", exist_ok=True)
csv_path = f"csvs/fric_sweep_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
log = []
header = [
    "t",
    "q","qd","qdd",
    "q_des","qd_des","qdd_des",
    "tau_cmd","tau_meas","tau_model",
    "tau_res","tau_hat_f",
    "theta1","theta2","theta3",
    "s"
]


# Helper: end-effector world pose
def ee_pose():
    pos, orn, _, _, _, _ = p.getLinkState(arm_id, EE_LINK, computeForwardKinematics=True)
    return pos, orn

# Compute Jacobian at a point on the fingertip (end of link2)
# link frame has COM at L2/2 along +X; fingertip is at local +X by another L2/2
LOCAL_TIP = [L2/2, 0, 0]  # relative to link2 COM


# Dynamics helpers

def get_jacobian(q, qd):
    Jlin, Jang = p.calculateJacobian(arm_id, EE_LINK, LOCAL_TIP, list(q), list(qd), [0.0, 0.0])
    J = np.array(Jlin)  # 3 x n
    return J  # linear Jacobian

def get_q_qd(arm_id):
    q = []; qd = []
    for j in range(DOF):
        js = p.getJointState(arm_id, j)
        q.append(js[0]); qd.append(js[1])
    return np.array(q), np.array(qd)

def mass_matrix_M(arm_id, q):
    # Full inertia matrix M(q)
    return np.array(p.calculateMassMatrix(arm_id, q))

def gravity_torque_G(arm_id, q):
    # G(q) = ID(q, 0, 0)
    zeros = [0.0]*DOF
    return np.array(p.calculateInverseDynamics(arm_id, q, zeros, zeros))

def coriolis_term_Cqd(arm_id, q, qd):
    # C(q,qd)qd = ID(q, qd, 0) - G(q)
    zeros = [0.0]*DOF
    tau_id = np.array(p.calculateInverseDynamics(arm_id, q, qd, zeros))
    return tau_id - gravity_torque_G(arm_id, q)

def modeled_torque_ID(arm_id, q, qd, qdd):
    # tau_model = M(q) qdd + C(q,qd) qd + G(q)
    return np.array(p.calculateInverseDynamics(arm_id, q, qd, qdd))

# -------------------------
# Adaptive control
# - 
# - 
# -------------------------
v_st   = 0.15      # Stribeck transition speed (rad/s) offline calculation
v_coul = 0.03      # Coulomb saturation speed (rad/s)   offline calculation
Lambda = 1.5       # tracking surface gain (>=0)
KDs    = 2.0       # feedback gain on s
Gamma_f = np.diag([0.005, 0.005, 0.001])  # adaptation gains to tune
# Gamma_f = np.zeros((3,3))
Gamma_eps = 0.0    # bias integrator increase to enable

# init parameters (from offline fit or small positive guesses)
# theta_f = np.array([0.05, 0.2, 0.03])  # [f_brk - f_c, f_c, f_vis]
theta_f = np.array([0.0, 0.0, 0.0])  # start neutral

eps = 0.0

def Yf1(v, v_st, v_coul):
    return np.array([
        np.exp(-(v / v_st)) * (v / v_st),
        np.tanh(v / v_coul),
        v
    ])  # shape (3,) but check error

def Yf2(v, v_st, v_coul, v_clip=2.0, eps=1e-6):
    v_st   = max(abs(v_st), eps)
    v_coul = max(abs(v_coul), eps)
    v_sat  = float(np.clip(v, -v_clip, v_clip))
    # use |v|/v_st in the Stribeck envelope; keep odd symmetry via the multiplier
    return np.array([
        np.exp(-(abs(v_sat) / v_st)) * np.sign(v_sat),   # static→coulomb envelope, odd
        np.tanh(v_sat / v_coul),                         # Coulomb smooth sign
        v_sat                                            # viscous
    ])

def step_adaptive(q, qd, q_des, qd_des, qdd_des, dt, tau_model, warmup):
    # tracking
    global theta_f, eps
    
    e  = q  - q_des
    ed = qd - qd_des
    if warmup:
        s = 0.0
    else:
        s  = ed + Lambda * e
        if abs(qd) < 0.05:
            s=0.0
        s = np.clip(s, -0.2, 0.2)


    # friction estimate
    phi = Yf2(qd, v_st, v_coul)         # (3,)
    tau_hat_f = float(phi @ theta_f)   # scalar
    tau_hat_f = float(np.clip(tau_hat_f, -2.5, 2.5))

    tau_hat_f =0.0 # TEST, DELETE LATER


    # tau_fb  = -KDs * s
    # tau_cmd = tau_model + tau_fb - tau_hat_f + eps
    tau_fb = Kp*(q_des - q) + Kd*(qd_des - qd) # SAFE PD CONTROLER
    tau_cmd = tau_model + tau_fb + tau_hat_f + eps

    # tau_cmd = float(np.clip(tau_cmd, -4.0, 4.0))


    # # theta_dot = -Gamma_f * phi^T * s
    # theta_update = - (Gamma_f @ (phi * s)) * dt  # broadcasts
    # theta_new = theta_f + theta_update
    
    if s != 0.0: # only learn when moving
        theta_new = theta_f - (Gamma_f @ (phi * s)) * dt
        theta_new[1] = max(theta_new[1], 0.0)  # f_c sempre positivo
        theta_new[2] = max(theta_new[2], 0.0)  # f_vis sempre positivo
        delta = np.clip(theta_new - theta_f, -0.02, 0.02)
        theta_f[:] = theta_f + delta

    # # bias integrator
    # if Gamma_eps > 0.0:
    #     eps += -Gamma_eps * s * dt

    return tau_cmd, tau_hat_f, s, theta_f.copy(), eps







# -------------------------
# Main loop
# - 
# - 
# -------------------------
err_int = 0.0
slide = True
qd_prev = np.zeros(2)
FS = 1/DT
CUTOFF = 5.0  # Hz

sos = butter(2, CUTOFF, btype='low', fs=FS, output='sos') 
zi_v = sosfilt_zi(sos); q_prev = p.getJointState(arm_id, 0)[0]

sos_a = butter(2, CUTOFF, btype='low', fs=FS, output='sos')
zi_a = sosfilt_zi(sos_a)

q_prev = 0.0
qd_f_prev = 0.0

DERIVATIVE_MODE = "butter"  # "NUM", "SAVGOL", "butter"

i = 0


p.changeVisualShape(arm_id, 1, rgbaColor=[1, 0, 0, 1]) # turn arm red
p.changeVisualShape(arm_id, 0, rgbaColor=[0, 1, 0, 1]) # turn arm green
set_camera_top()
print("Running")

try:
    print("Friction test moving joint 0 slow")

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


        t = time.time() - t0
        if t > RUN_SECS:
            break

        # FREEZE JOINT 1
        p.setJointMotorControl2(arm_id, 1, p.POSITION_CONTROL, targetPosition=0.0, force=100, positionGain=1, velocityGain=1.0)

        # Desired traject joint 0; joint 1 stays still fah now
        q0_des  = A * math.sin(w*t)
        qd0_des = A*w * math.cos(w*t)
        qdd0_des = -A*w*w * math.sin(w*t)

        # Read state
        q_vec  = [p.getJointState(arm_id, j)[0] for j in (0,1)]
        qd_vec = [p.getJointState(arm_id, j)[1] for j in (0,1)]

        q_hist.append(q_vec[JIDX])
        t_hist.append(t)


        # ----------- SAFE BASELINE CONTROL --------------
        q0 = q_vec[0]
        qd0 = qd_vec[0]

        # gravity only (no C, no M yet)
        G = np.array(p.calculateInverseDynamics(arm_id, [q0, q_vec[1]], [0.0,0.0], [0.0,0.0]))
        G0 = float(G[0])

        #----------------GHOST ARM--------------------
        # Pose ghost arm at desired trajectory (sinusoidal reference)
        p.resetJointState(ghost_id, 0, q0_des, qd0_des)
        p.resetJointState(ghost_id, 1, 0.0, 0.0)


        # # Gravity G(q)  [2] with q=0 qdot=0
        # zeros = [0.0, 0.0]
        # g = np.array(p.calculateInverseDynamics(arm_id, q_vec, zeros, zeros))
        # Cqd = np.array(p.calculateInverseDynamics(arm_id, q_vec, qd_vec, zeros)) - g


        # tau_model = np.array(p.calculateInverseDynamics(arm_id, q_vec.tolist(), qd_vec.tolist(), qdd_vec.tolist()))
        
        # CHECK IF WINDOW IS FILLED
        have_window = (len(q_hist) == WIN)
        
        
        if DERIVATIVE_MODE == "NUM":
            qdd_vec = [(qd_vec[i] - qd_prev[i]) / DT for i in range(2)]
            qd_prev = qd_vec
            tau_model = np.array(p.calculateInverseDynamics(arm_id, q_vec, qd_vec, qdd_vec))
        
        elif DERIVATIVE_MODE == "SAVGOL":
            if have_window:
                q_arr   = np.array(q_hist)             # history of joint 0 position
                q0_f    = savgol_filter(q_arr, WIN, POLY, deriv=0)[-1]
                qd0_f   = savgol_filter(q_arr, WIN, POLY, deriv=1, delta=DT)[-1]
                qdd0_f  = savgol_filter(q_arr, WIN, POLY, deriv=2, delta=DT)[-1]
            else:
                q0_f, qd0_f, qdd0_f = q_vec[0], qd_vec[0], 0.0
            
            q1_f, qd1_f, qdd1_f = q_vec[1], 0.0, 0.0  # joint 1 fixed 
            qdd_vec = [qdd0_f, qdd1_f]
            tau_model = np.array(p.calculateInverseDynamics(
                arm_id,
                [q0_f,  q1_f],
                [qd0_f, qd1_f],
                [qdd0_f,qdd1_f]
            ))
            tau_model0 = float(tau_model[JIDX])

        elif DERIVATIVE_MODE == "butter":
            qd_fd = (q_vec[0] - q_prev)/DT
            qd0_f, zi_v = sosfilt(sos, [qd_fd], zi=zi_v)
            q_prev = q_vec[0]
            qd0_f = float(qd0_f[0])

            qdd_fd = (qd0_f - qd_f_prev) / DT
            qd_f_prev = qd0_f   

            qdd_f_arr, zi_a = sosfilt(sos_a, [qdd_fd], zi=zi_a)
            qdd0_f = float(qdd_f_arr[0])
            tau_model = np.array(p.calculateInverseDynamics(
                arm_id,
                [q_vec[0],  0.0],
                [qd0_f, 0.0],
                [qdd0_f,0.0]
            ))
            tau_model0 = float(tau_model[0])
            


        tau1_cmd = 0.0  # joint 2 no friction ID for now

        # ------------- SAFE BASELINE CONTROL -------------
        tau_fb_safe = Kp*(q0_des - q_vec[0]) + Kd*(qd0_des - qd0_f)

        tau_model_safe = np.array(p.calculateInverseDynamics(
            arm_id,
            [q0,  q_vec[1]],
            [qd0, 0.0],
            [qdd0_des, 0.0]  
        ))
        tau_model0_safe = float(tau_model_safe[0])

        tau0_cmd_safe = tau_model0_safe + tau_fb_safe
        tau0_cmd_safe = float(np.clip(tau0_cmd_safe, -8.0, 8.0))
        tau_model0 = tau_model0_safe

        
        # --- call adaptive using FILTERED q, qd ---

        speed_ok = abs(qd0_f) > 0.05     # avoid reversals/standstill
        time_ok  = t > 0.5               # small startup delay
        ready = have_window and speed_ok and time_ok

        if not ready and DERIVATIVE_MODE == "SAVGOL":
           # WARM-UP: PD + model ONLY (no adaptive, no θ update)
            tau_fb  = Kp*(q0_des - q0) + Kd*(qd0_des - qd0_f)
            tau0_cmd = tau_model0 + tau_fb
            tau0_hat_f = 0.0
            s0 = 0.0
            theta_snapshot = theta_f.copy()   # unchanged 
            
        else:
            # print("Ready: adaptive control active")
            tau0_cmd, tau0_hat_f, s0, theta_snapshot, eps_now = step_adaptive(
                q=q_vec[0], qd=qd0_f,
                q_des=q0_des, qd_des=qd0_des, qdd_des=qdd0_des,
                dt=DT, tau_model=tau_model0,
                warmup=False
            )
        # tau0_cmd = float(np.clip(tau0_cmd, -4.0, 4.0))  # limit torque

        
    
        # Apply torques
        # p.setJointMotorControlArray(arm_id, [0,1], p.TORQUE_CONTROL, forces=[tau0_cmd, tau1_cmd])
        # p.setJointMotorControlArray(arm_id, [0,1], p.TORQUE_CONTROL, forces=[tau0_cmd_safe, 0.0])
        #APPLY TORQUE WITH KNOWN FRICTION MODEL
        # tau_applied = tau0_cmd - tau_f_simple(qd_vec[0])   # friction opposes motion
        tau_applied = tau0_cmd_safe - tau_f_simple(qd_vec[0])   # SAFER VERSION PLS
        tau_applied = float(np.clip(tau_applied, -8, 8))
        p.setJointMotorControl2(arm_id, 0, p.TORQUE_CONTROL, force=tau_applied) # ONLY 1 joint

        print("TORQUE SAFE:", tau0_cmd_safe, "TORQUE ADAPTIVE:", tau0_cmd, "TORQUE APPLIED:", tau_applied)
        
        p.stepSimulation()
        time.sleep(DT)

        tau_meas0 = tau0_cmd
        tau_res0  = tau_meas0 - tau_model0

        
        if not(i % 100) and i>0:
            log.append([
                t,
                float(q_vec[JIDX]), float(qd0_f), float(qdd0_f),
                q0_des, qd0_des, qdd0_des,
                float(tau0_cmd), float(tau_meas0), float(tau_model0),
                float(tau_res0), float(tau0_hat_f),
                float(theta_snapshot[0]), float(theta_snapshot[1]), float(theta_snapshot[2]),
                float(s0)
            ])
        # print("torque1:" , p.getJointState(arm_id, 0)[3], "torque2:", p.getJointState(arm_id, 1)[3])
        # print("tau1:" , tau1, "tau_meas:", tau1_meas, "tau_model:", tau_model[0], "tau_res:", tau1_res)
        i += 1

    # Save CSV
    os.makedirs("csvs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = f"csvs/friction_run_joint1_{timestamp}.csv"

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(log)
    print(f"Wrote {out} with {len(log)} rows.")

except KeyboardInterrupt:
    pass
