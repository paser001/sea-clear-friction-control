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
L1, L2 = 0.5, 0.5          # link lengths
M1= 2.0
M2= 2.0
TMAX = 2.0
RUN_SECS = 120.0
RUN1_SECS = 120.0
RUN2_SECS = 420.0
RUN3_SECS= 480.0
RUN4_SECS = 540.0
RUN5_SECS = 500.0
ONCE = True

DT = 1.0/240.0
DOF = 1               
JIDX = 0                  # identify friction on joint 1
OTHER = 1
t0 = time.time()
log = []
WIN = 31         # window size (gotta be odd)
POLY = 3         # polynomial order filtre

START_POS = [0 , 0]

q_hist  = deque(maxlen=WIN)
t_hist  = deque(maxlen=WIN)


EE_LINK = 1  
LOCAL_TIP = [L2/2, 0, 0]  # fingertip position in link2 frame


# ---------- True friction (plant) ----------
# Stribeck: tau_f(qd) = Fc*sign(qd) + Fv*qd + (Fs-Fc)*exp( -( |qd|/vs )^2 )
Fc_true = 0.1     # Nm  (Coulomb)
Fs_true = 0.12     # Nm  (static peak)
Fv_true = 0.02     # Nms/rad (viscous)
vs_true = 0.10     # rad/s  (Stribeck speed)




noise_sigma = 0.01 # Nm (torque noise)
Fc_simple = 0.15      # Coulomb friction [Nm]
Fv_simple = 0.02      # viscous friction [Nms/rad]
v_eps_simple = 0.02   # smoothing speed [rad/s]


# ----- Plateau config -----
q_min = -1.8   # rlower bound
q_max =  1.8   # upper bound

# plateau_speeds = [0.3,0.5, 0.8, 1.6, 2.0]  # rad.sec^-1

plateau_speeds = [0.3,0.5, 0.8, 1.6, 2.0, 2.5, 2.2, 2.8, 3.0, 4.0, 4.5, 3.5, 1.7, 1.5, 1.3, 1.0]  # rad.sec^-1

plateau_speeds.reverse()  # start with fast speeds
plateau_time   = 8.0               # seconds per plateau
bounce_timer = 0.0


# PLATEU 2
plateau_segments = [
    (-0.8, -0.4, 0.2),
    (-0.4,  0.0, 0.3),
    ( 0.0,  0.4, 0.4),
    ( 0.4,  0.8, 0.5),
]

seg_idx  = 0
dir_sign = +1   # +1 going forward through segments, -1 backward

# State for excitation
exc_idx   = 0          # plateau_seq index
exc_t0    = 0.0        # start time current plateau
v_ref     = 0.0        # current cmd joint velocity
q0_des    = 0.0        # desired joint position


# -------------------------
# Adaptive control
# - 
# - 
# -------------------------

v_st   = 0.1     # Stribeck transition speed (rad/s) offline calculation
v_coul = 0.07      # Coulomb saturation speed (rad/s)   offline calculation



# Gamma_f = np.diag([0.01, 0.05, 0.01])  # adaptation gains to tune
Gamma_f = np.diag([0.4, 0.1, 0])  # adaptation gains to tune
Gamma_f_fast = np.diag([Gamma_f[0,0],Gamma_f[1,1],0.1])
# Gamma_f = np.zeros((3,3))
Gamma_eps = 0.1    # bias integrator increase to enable


 
adapt_freeze_timer = 0.0



Kp, Kd = 1, 0.5   # KD= 5 is good for adaptive on
Kd_s =5
# Lambda = Kp / Kd_s
Lambda = 1.2     # 0.5 is good for adapative on

# tau_fb  = -Kd_s * s
# s  = edot + Lambda * e

# init parameters offline fit or small positive guesses
# theta_f = np.array([0.01, 0.01, 0.005]) # MEH START
# theta_f = np.array([0.015, 0.08, 0.015])
# theta_f = np.array([0.0001, 0.0001, 0.0001])  # start neutral 
theta_f = np.array([0.015, 0.1, 0.025])
s_clip = 0.5
tau_hat_f_clip = 8.0

ADAPTATION = True

eps = 0.0

# MAIN LOOP
# TEST_MODE = "SIN"  # "PLATEAUS", "SIN"
CUSTOM_TRIAL = False

TEST_MODE = "SIN"

A = math.radians(90)    # amplit
w = 0.2                 # rad/s


err_int = 0.0
slide = True
qd_prev = np.zeros(2)
qd0_prev = 0.0
FS = 1/DT
CUTOFF = 25.0  # Hz for filtering

q_prev = 0.0
qd_f_prev = 0.0

Kd_warmup = 8.0
Kp_warmup = 0.4

DERIVATIVE_MODE = "butter"  # "NUM", "SAVGOL", "butter"


# -------------------------
# Start PyBullet
# -------------------------
cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81) # WITH GRAVITY
# p.setGravity(0,0,0)      # NO GRAVITY
p.setTimeStep(DT)

#contacts frictionmaxxin and stable
p.setPhysicsEngineParameter(numSolverIterations=200, numSubSteps=2, contactERP=0.3, frictionERP=0.3)
p.loadURDF("plane.urdf")

def set_camera_top():
    p.resetDebugVisualizerCamera(cameraDistance=2.0,
                                 cameraYaw=90,
                                 cameraPitch=-89,
                                 cameraTargetPosition=[1, 0.1, 0.05])

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
# TODO CHANGE ANGLE LIMITES
urdf = f"""<?xml version="1.0"?>
<robot name="finger2r">
  <link name="base"/>
  <link name="link1">
    <inertial><origin xyz="{L1/2} 0 0"/><mass value="2.0"/><inertia ixx="0.0417" iyy="0.0417" izz="0.0417"/></inertial>
    <visual><origin xyz="{L1/2} 0 0"/><geometry><box size="{L1} 0.08 0.08"/></geometry></visual>
    <collision><origin xyz="{L1/2} 0 0"/><geometry><box size="{L1} 0.08 0.08"/></geometry></collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <dynamics damping="0.0" friction="0.0"/>
    <limit lower="-300.14" upper="300.14" effort="50" velocity="6.0"/>
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




# Disable default motor to command torques
for j in range(DOF):
    p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)

# Move to an angle positon over some steps
def goto(id, q1, q2, steps=480):
    p.setJointMotorControl2(arm_id, 0, p.POSITION_CONTROL,
                            targetPosition=q1, force=20,
                            positionGain=0.1, velocityGain=1.0)
    for _ in range(steps):
        p.stepSimulation(); time.sleep(DT)
    # release motors for torque control
    for j in range(DOF):
        p.setJointMotorControl2(id, j, p.VELOCITY_CONTROL, force=0)


def tau_f_simple(qd):
    return Fc_simple * np.tanh(qd / v_eps_simple) + Fv_simple * qd


def tau_f_true(qd, simple = False):
    if simple:
        return tau_f_simple(qd)
    else:
        mag = Fc_true + (Fs_true - Fc_true) * np.exp(-(abs(qd)/vs_true)**2)
        return mag * np.sign(qd) + Fv_true * qd



# Build sequence: +v, -v for each speed
def plateau_builder():
    plateau_seq = []
    for v in plateau_speeds:
        plateau_seq.append(+v)
        plateau_seq.append(-v)
    return plateau_seq


# ---------- Disable default motors, center pose ----------

for j in range(DOF):
    p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)
for j in range(DOF):
    p.setJointMotorControl2(ghost_id, j, p.VELOCITY_CONTROL, force=0)
# Initial pose: reach to near wall # WALL IS GONE :(
# goto(q1=-0.59695293, q2=-0.39810182, steps=1)
goto(arm_id, q1=START_POS[0], q2=START_POS[1])
goto(ghost_id, q1=START_POS[0], q2=START_POS[1])
sleep_temp(240)
for j in range(DOF): 
    p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)


# ---------- Logging setup ----------

os.makedirs("csvs", exist_ok=True)
csv_path = f"csvs/fric_sweep_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
log = []
header = [
    "t",
    "q","qd_f","qdd_f",
    "q_des","qd_des","qdd_des",
    "tau_cmd","tau_model",
    "tau_res","tau_hat_f", "tau_f_true","tau_f_err",
    "theta1","theta2","theta3",
    "s", "eps", "tau_fb"
]

def log_marker_row(t):
    N_COLS = len(header)
    return [t] + [float('nan')] * (N_COLS - 1)



# Helper: end-effector world pose
def ee_pose():
    pos, orn, _, _, _, _ = p.getLinkState(arm_id, EE_LINK, computeForwardKinematics=True)
    return pos, orn

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
    return np.array(p.calculateInverseDynamics(arm_id, q, qd, qdd))



def Yf1(v, v_st, v_coul): # OVERFLOWS QUITE OFTEN
    return np.array([
        np.exp(-(v / v_st)) * (v / v_st),
        np.tanh(v / v_coul),
        v
    ])  

def Yf2(v, v_st, v_coul, v_clip=5.0, eps=1e-6):  # SAFE CHAT GPT VERSION
    v_st   = max(abs(v_st), eps)
    v_coul = max(abs(v_coul), eps) # these 2 kinda useless tbh they re fixed parameters

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


    # AVOID LEARNING NEAR JOINT LIMITS

    q_next = q + qd * dt   # 1-step prediction (good enough)
    margin = 0.2



    near_limit_now  = (q <= q_min + margin) or (q >= q_max - margin)
    will_hit_limit  = (q_next <= q_min + margin) or (q_next >= q_max - margin)

    adapt_ok = (not near_limit_now) and (not will_hit_limit)


    if will_hit_limit:
        adapt_freeze_timer = 0.2  # seconds

    # if adapt_freeze_timer > 0:
    #     adapt_ok = False
    #     adapt_freeze_timer -= dt


    # print("e:", e, "ed:", ed)

    if warmup:
        s = 0.0
    else:
        s  = ed + Lambda * e
        s_adapt = s
        s = np.clip(s, -2.5, 2.5)
        # s_adapt = np.clip(s, -s_clip, s_clip)
        

    # print("s:", s)

    # friction estimate
    phi = Yf2(qd, v_st, v_coul)         # (3,)
    tau_hat_f = float(phi @ theta_f)   # scalar
    tau_hat_f = float(np.clip(tau_hat_f, -tau_hat_f_clip, tau_hat_f_clip))

    # tau_hat_f =0.0 # TEST, DELETE LATER
    # tau_model = 0.0 # TEST, DELETE LATER

    


    tau_fb  = -Kd_s * s
    # tau_fb = Kp*(q_des - q) + Kd*(qd_des - qd) # SAFE PD CONTROLER
    tau_fb = float(np.clip(tau_fb, -10.5, 10.5))
    tau_cmd = tau_model + tau_fb + tau_hat_f + eps

    # if abs(s)>2.5:
    #     tau_cmd =0.0 # TEST, DELETE LATER


    # # theta_dot = -Gamma_f * phi^T * s
    # theta_update = - (Gamma_f @ (phi * s)) * dt  # broadcasts
    # theta_new = theta_f + theta_update

    
    
    if abs(s)>0.0 and abs(s_adapt)<0.5 and abs(qd)>0.1 and ADAPTATION == True and adapt_ok : # only learn when moving
        if abs(qd)<1.5:
            # theta_new = theta_f - ([Gamma_f[0],Gamma_f[1],0.0] @ (phi * s_adapt)) * dt
            theta_new = theta_f - (Gamma_f @ (phi * s_adapt)) * dt
            print("Adapting (low speed)")
        else:
            theta_new = theta_f - (Gamma_f_fast @ (phi * s_adapt)) * dt
        theta_new[1] = max(theta_new[1], 0.0)  # f_c sempre positivo
        theta_new[2] = max(theta_new[2], 0.0)  # f_vis sempre positivo
        delta = np.clip(theta_new - theta_f, -0.05, 0.05) # TO TUNE
        theta_f[:] = theta_f + delta
        print("Theta updated:", theta_f)

    # bias integrator
    if Gamma_eps > 0.0:
        eps += -Gamma_eps * s * dt

    return tau_cmd, tau_hat_f, s, theta_f.copy(), eps



# -------------------------
# Main loop
# - 
# - 
# -------------------------


sos = butter(2, CUTOFF, btype='low', fs=FS, output='sos') 
zi_v = sosfilt_zi(sos)
q_prev = p.getJointState(arm_id, 0)[0]

sos_a = butter(2, CUTOFF, btype='low', fs=FS, output='sos')
zi_a = sosfilt_zi(sos_a)


i = 0

# ----- Plateau excitation state -----
plateau_seq = plateau_builder()      # build once
exc_idx = 0                          # which plateau (index in plateau_seq)
v_ref = 0.0                          # filtered/commanded velocity
q0_des = p.getJointState(arm_id, 0)[0]  # start from current q0

t_sim = 0.0                          # simulation time (integrated, not wall-clock)
t_plateau = 0.0                      # time spent in current plateau
alpha = 2.0 * DT                     # ramp factor for v_ref

qd_des_prev = 0.0



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
            goto(arm_id, q1=-0.59695293, q2=-0.39810182, steps=240)
            err_int = 0.0
            i = 0
            sleep_temp(240)
            print("Reset done")
        if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
            break


        t = time.time() - t0
        if CUSTOM_TRIAL:
            if t > RUN1_SECS and t < RUN2_SECS:
                TEST_MODE = "PLATEAUS"
            elif t > RUN2_SECS and t < RUN3_SECS:
                w=2.5
                TEST_MODE = "SIN"
            elif t > RUN3_SECS and t < RUN4_SECS:
                w=1.5
                TEST_MODE = "SIN"
            elif t > RUN4_SECS:
                TEST_MODE = "PLATEAUS"

        if t> RUN5_SECS and ONCE:
            # Fc_true = 0.2     # Nm  (Coulomb)
            # Fs_true = 0.24     # Nm  (static peak)
            # Fv_true = 0.04     # Nms/rad (viscous)
            # vs_true = 0.20     # rad/s  (Stribeck speed)
            # exc_idx    = 0
            # t_plateau  = 0.0
            # v_ref      = 0.0
            # q0_des     = q_vec[0]
            q_min = -100.8   # rlower bound
            q_max =  100.8   # upper bound
            ADAPTATION = False
            log.append(log_marker_row(t))
            ONCE = False
        
        if t > RUN_SECS:
            break

        # FREEZE JOINT 1
        # p.setJointMotorControl2(arm_id, 1, p.POSITION_CONTROL, targetPosition=0.0, force=100, positionGain=1, velocityGain=1.0)


        if TEST_MODE == "SIN":
            # Desired traject joint 0; joint 1 stays still fah now
            q0_des  = A * math.sin(w*t)
            qd0_des = A*w * math.cos(w*t)
            qdd0_des = -A*w*w * math.sin(w*t)
        elif TEST_MODE == "PLATEAUS":
            t_sim += DT
            t_plateau += DT

            # if plateau time elapsed, switch to next plateau speed
            if t_plateau > plateau_time:
                exc_idx = (exc_idx + 1) % len(plateau_seq)
                t_plateau = 0.0
                print("v_tgt =", plateau_seq[exc_idx])

            v_tgt = plateau_seq[exc_idx]

            # smooth ramp to target velocity
            v_ref = v_ref + alpha * (v_tgt - v_ref)

            # integrate velocity to get desired position
            q0_des = q0_des + v_ref * DT

            
            if q0_des > q_max:
                q0_des = q_max
                v_ref  = -abs(v_ref)
                exc_idx = (exc_idx + 1) % len(plateau_seq)  # flip sign immediately
                t_plateau = 0.0             # restart plateau timer on bounce
            elif q0_des < q_min:
                q0_des = q_min
                v_ref  = +abs(v_ref)
                exc_idx = (exc_idx + 1) % len(plateau_seq)  # flip sign immediately
                t_plateau = 0.0

            qd0_des  = v_ref
            qdd0_des = (qd0_des - qd_des_prev) / DT  # approximate acceleration
            qd_des_prev = qd0_des

        elif TEST_MODE == "PLATEAUS2":
            # current segment info
            q_lo, q_hi, v_mag = plateau_segments[seg_idx]
            v_tgt = dir_sign * v_mag

            # smooth ramp in velocity
            v_ref = v_ref + alpha * (v_tgt - v_ref)

            # integrate to get desired position
            q0_des = q0_des + v_ref * DT

            # check if we left the current [q_lo, q_hi] window
            left_segment = (q0_des < q_lo) or (q0_des > q_hi)

            if left_segment:
                # clamp back inside
                q0_des = min(max(q0_des, q_lo), q_hi)

                # advance / rewind segment index
                seg_idx += dir_sign
                if seg_idx >= len(plateau_segments):
                    seg_idx = len(plateau_segments) - 1
                    dir_sign = -1    # bounce and go backwards
                elif seg_idx < 0:
                    seg_idx = 0
                    dir_sign = +1    # bounce and go forwards

                # recompute new target speed for the new segment
                q_lo, q_hi, v_mag = plateau_segments[seg_idx]
                v_tgt = dir_sign * v_mag
                v_ref = v_tgt   # or keep smooth ramp if you prefer

            qd0_des  = v_ref
            qdd0_des = 0.0



        # Read state
        q_vec  = [p.getJointState(arm_id, j)[0] for j in range(DOF)]
        qd_vec = [p.getJointState(arm_id, j)[1] for j in range(DOF)]

        q_hist.append(q_vec[JIDX])
        t_hist.append(t)


        # ----------- SAFE BASELINE CONTROL --------------
        q0 = q_vec[0]
        qd0 = qd_vec[0]

        # gravity only (no C, no M yet)
        # G = np.array(p.calculateInverseDynamics(arm_id, [q0, q_vec[1]], [0.0,0.0], [0.0,0.0]))
        # G0 = float(G[0])

        #----------------GHOST ARM--------------------
        # Pose ghost arm at desired trajectory (sinusoidal reference)
        p.resetJointState(ghost_id, 0, q0_des, qd0_des)


        # # Gravity G(q)  [2] with q=0 qdot=0
        # zeros = [0.0, 0.0]
        # g = np.array(p.calculateInverseDynamics(arm_id, q_vec, zeros, zeros))
        # Cqd = np.array(p.calculateInverseDynamics(arm_id, q_vec, qd_vec, zeros)) - g


        # tau_model = np.array(p.calculateInverseDynamics(arm_id, q_vec.tolist(), qd_vec.tolist(), qdd_vec.tolist()))
        
        # CHECK IF WINDOW IS FILLED
        have_window = (len(q_hist) == WIN)
        
        
        if DERIVATIVE_MODE == "NUM":
            qdd_vec = (qd0 - qd0_prev) / DT
            qd0_prev = qd0
            qdd0_f = qdd_vec
            qd0_f = qd_vec[0]
            q0_f = q0
            tau_model = np.array(p.calculateInverseDynamics(
                arm_id,
                [q0],
                [qd0_f],
                [qdd0_f]
            ))
            tau_model0 = float(tau_model[0])

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

            # TEST TEST TEST
            # qd0_f = qd0
            # qdd0_f = qdd0_des
            # qdd0_f = 0.0
            # q_prev= q0

            tau_model = np.array(p.calculateInverseDynamics(
                arm_id,
                [q0],
                [qd0_f],
                [qdd0_des] # CHANGED TO DESIRED
            ))
            tau_model0 = float(tau_model[0])
    
            


        tau1_cmd = 0.0  # joint 2 no friction ID for now

        # ------------- SAFE BASELINE CONTROL -------------
        # tau_fb_safe = Kp*(q0_des - q_vec[0]) + Kd*(qd0_des - qd0_f)

        # tau_model_safe = np.array(p.calculateInverseDynamics(
        #     arm_id,
        #     [q0,  q_vec[1]],
        #     [qd0, 0.0],
        #     [qdd0_des, 0.0]  
        # ))
        # tau_model0_safe = float(tau_model_safe[0])

        # tau0_cmd_safe = tau_model0_safe + tau_fb_safe
        # tau0_cmd_safe = float(np.clip(tau0_cmd_safe, -8.0, 8.0))
        # tau_model0 = tau_model0_safe

        # --- call adaptive using FILTERED q, qd ---

        speed_ok = abs(qd0) > 0.05     # avoid reversals/standstill
        time_ok  = t > 0.5               # small startup delay
        ready = have_window and speed_ok and time_ok

        if not ready and DERIVATIVE_MODE == "SAVGOL":
           # WARM-UP: PD + model ONLY (no adaptive, no θ update)
            tau_fb  = Kp_warmup*(q0_des - q0) + Kd_warmup*(qd0_des - qd0_f)
            tau0_cmd = tau_model0 + tau_fb
            tau0_hat_f = 0.0
            s0 = 0.0
            theta_snapshot = theta_f.copy()   # unchanged 
            
        else:
            # print("Ready: adaptive control active")
            q_des_vec = [p.getJointState(ghost_id, j)[0] for j in range(DOF)]
            qd_des_vec = [p.getJointState(ghost_id, j)[1] for j in range(DOF)]

            qd_des_f = qd_des_vec[0]  # des velo
            q_des_f = q_des_vec[0]  # desired pos
            # print("qd:", qd0_des, "qd error", qd0_f - qd_des_f)
            # print("q error", q0 - q_des_f)

            tau0_cmd, tau0_hat_f, s0, theta_snapshot, eps_now = step_adaptive(
                q=q_vec[0], qd=qd0_f,
                q_des=q_des_f, qd_des=qd_des_f, qdd_des=qdd0_des,
                dt=DT, tau_model=tau_model0,
                warmup=False
            )
    
        # Apply torques
        # p.setJointMotorControlArray(arm_id, [0,1], p.TORQUE_CONTROL, forces=[tau0_cmd, tau1_cmd])
        # p.setJointMotorControlArray(arm_id, [0,1], p.TORQUE_CONTROL, forces=[tau0_cmd_safe, 0.0])
        #APPLY TORQUE WITH KNOWN FRICTION MODEL
        tau_f_true_val = tau_f_true(qd0_f, simple = False)
        tau_applied = tau0_cmd - tau_f_true_val   # friction opposes motion 
        # tau_applied = tau0_cmd_safe - tau_f_simple(qd_vec[0])   # SAFER VERSION PLS
        # tau_applied = tau0_cmd #TEST PD KDS CONTROLLER
        tau_applied = float(np.clip(tau_applied, -8, 8))
        p.setJointMotorControl2(arm_id, 0, p.TORQUE_CONTROL, force=tau_applied) # ONLY 1 joint
        # p.setJointMotorControl2(arm_id, 0, p.TORQUE_CONTROL, 2) # TEST 2Nm
        
        p.stepSimulation()
        time.sleep(DT)

        tau_meas0 = tau0_cmd
        tau_res0  = tau_meas0 - tau_model0

        
        if not(i % 100) and i>0:

            log.append([
                t,
                float(q_vec[JIDX]), float(qd0_f), float(qdd0_f),
                q0_des, qd0_des, qdd0_des,
                float(tau0_cmd), float(tau_model0),
                float(tau_res0), float(tau0_hat_f), float(tau_f_true_val), float(abs(tau_f_true_val - tau0_hat_f)),
                float(theta_snapshot[0]), float(theta_snapshot[1]), float(theta_snapshot[2]),
                float(s0), float(eps_now), float(tau0_cmd - tau_model0 - tau0_hat_f)
            ])
        # print("torque1:" , p.getJointState(arm_id, 0)[3], "torque2:", p.getJointState(arm_id, 1)[3])
        # print("tau1:" , tau1, "tau_meas:", tau1_meas, "tau_model:", tau_model[0], "tau_res:", tau1_res)
        print(f"({t}/{RUN_SECS})")
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
