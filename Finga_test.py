# =========================
# Friction-ID sweep loop
# =========================
import os, csv, math, time
import numpy as np
from datetime import datetime

import pybullet as p
import pybullet_data as pd

DT = 1.0/240.0            # keep consistent with p.setTimeStep
DOF = 2
JIDX = 0                  # identify friction on joint 1
OTHER = 1

# ---------- Hidden "true" friction (plant) ----------
# Stribeck: tau_f(qd) = Fc*sign(qd) + Fv*qd + (Fs-Fc)*exp( -( |qd|/vs )^2 )
Fc_true = 0.25     # Nm  (Coulomb)
Fs_true = 0.35     # Nm  (static peak)
Fv_true = 0.04     # Nms/rad (viscous)
vs_true = 0.20     # rad/s  (Stribeck speed)
noise_sigma = 0.01 # Nm (torque noise)

def tau_f_true(qd):
    return (Fc_true*np.sign(qd)
            + Fv_true*qd
            + (Fs_true - Fc_true)*np.exp(-(abs(qd)/vs_true)**2))

# ---------- Dynamics helpers (no friction) ----------
def get_q_qd(arm_id):
    q = []; qd = []
    for j in range(DOF):
        js = p.getJointState(arm_id, j)
        q.append(js[0]); qd.append(js[1])
    return np.array(q), np.array(qd)

def M_matrix(arm_id, q):
    return np.array(p.calculateMassMatrix(arm_id, q))

def G_torque(arm_id, q):
    zeros = [0.0]*DOF
    return np.array(p.calculateInverseDynamics(arm_id, q, zeros, zeros))

def Cqd_term(arm_id, q, qd):
    zeros = [0.0]*DOF
    id_q_qd = np.array(p.calculateInverseDynamics(arm_id, q, qd, zeros))
    return id_q_qd - G_torque(arm_id, q)

def tau_model_ID(arm_id, q, qd, qdd):
    return np.array(p.calculateInverseDynamics(arm_id, q, qd, qdd))

# ---------- Velocity plateau generator ----------
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

# ---------- Simple PD on position + velocity tracking ----------
# We integrate v_ref to a position reference q_ref to avoid bias.
Kp = 2.0
Kd = 0.05

# Ensure gravity is what you want for the model (0 or -9.81)
# p.setGravity(0, 0, 0)    # (recommended) isolate friction
# p.setGravity(0, 0, -9.81)  # include G(q) if you want to validate gravity modeling

# ---------- Disable default motors, center pose ----------
for j in (0,1):
    p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)
# move to a neutral pose then release again if you like:
# (optional) goto(q1=0.0, q2=0.0, steps=240)
# for j in (0,1): p.setJointMotorControl2(arm_id, j, p.VELOCITY_CONTROL, force=0)

# ---------- Logging setup ----------
os.makedirs("csvs", exist_ok=True)
csv_path = f"csvs/fric_sweep_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
log = []
header = [
    "t",
    "q1","q2","qd1","qd2","qdd1","qdd2",
    "q1_ref","qd1_ref",
    "tau_cmd1","tau_apply1","tau_cmd2","tau_apply2",
    "G1","G2","Cqd1","Cqd2","tau_model1","tau_model2",
    "tau_res1","tau_res2",
    "M11","M12","M21","M22",
    "v_target"
]

# ---------- Main sweep ----------
plateaus = schedule_plateaus()
plateau_idx = 0
hold_time = 0.0
v_ref = 0.0
q_ref = 0.0

# for finite-difference acceleration:
q_prev, qd_prev = get_q_qd(arm_id)

t0 = time.time()
print("[fric] Starting constant-velocity plateaus…")
while p.isConnected() and plateau_idx < len(plateaus):

    # time
    t = time.time() - t0

    # target speed for this plateau
    v_target = plateaus[plateau_idx]

    # smooth ramp of v_ref toward v_target
    v_ref = v_ref + alpha*(v_target - v_ref)
    # integrate to position reference to give Kp something meaningful
    q_ref = q_ref + v_ref*DT

    # read state
    q, qd = get_q_qd(arm_id)

    # PD torque for joint 1 only (joint 2 off)
    e_pos = q_ref - q[JIDX]
    e_vel = v_ref  - qd[JIDX]
    tau_cmd1 = Kp*e_pos + Kd*e_vel
    tau_cmd2 = 0.0

    # ---- Hidden plant friction injection on joint 1 ----
    # We apply what the *motor output* minus internal friction + noise would produce at the shaft.
    fric = tau_f_true(qd[JIDX])
    noise = np.random.normal(0.0, noise_sigma)
    tau_apply1 = tau_cmd1 - fric + noise
    tau_apply2 = tau_cmd2  # keep other joint off

    # Apply torques to Bullet
    p.setJointMotorControlArray(arm_id, [JIDX, OTHER], p.TORQUE_CONTROL,
                                forces=[tau_apply1, tau_apply2])

    # step
    p.stepSimulation()
    time.sleep(DT)

    # compute model terms (no friction) for residual
    # qdd via finite difference of qd
    q_now, qd_now = get_q_qd(arm_id)
    qdd = (qd_now - qd_prev) / DT

    G = G_torque(arm_id, q_now)
    Cqd = Cqd_term(arm_id, q_now, qd_now)
    tau_model = tau_model_ID(arm_id, q_now, qd_now, qdd)

    M = M_matrix(arm_id, q_now)  # 2x2

    # Measured motor torque for ID = what the motor *commands*
    tau_meas1 = tau_cmd1
    tau_meas2 = tau_cmd2

    tau_res = np.array([tau_meas1, tau_meas2]) - tau_model

    # log row
    log.append([
        t,
        q_now[0], q_now[1],
        qd_now[0], qd_now[1],
        qdd[0], qdd[1],
        q_ref, v_ref,
        tau_cmd1, tau_apply1, tau_cmd2, tau_apply2,
        G[0], G[1], Cqd[0], Cqd[1], tau_model[0], tau_model[1],
        tau_res[0], tau_res[1],
        M[0,0], M[0,1], M[1,0], M[1,1],
        v_target
    ])

    # roll fd states
    qd_prev = qd_now

    # plateau timing: when v_ref is close to v_target, count hold time
    if abs(v_ref - v_target) < 0.02*max(0.05, abs(v_target)):
        hold_time += DT
    else:
        hold_time = 0.0

    # advance plateau when held long enough
    if hold_time >= plateau_hold:
        plateau_idx += 1
        hold_time = 0.0

# write CSV
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(log)

print(f"[fric] Done. Saved {len(log)} rows to {csv_path}")
