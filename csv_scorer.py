import numpy as np
import pandas as pd
from pathlib import Path

folder = Path("./csvs")
csv_files = list(folder.glob("*.csv"))


def analyze(path):
    df = pd.read_csv(path)
    mask = df["t"] > 5.0
    mask_fric = mask & (np.abs(df["qd"]) > 0.05)

    # tracking
    e_q = df["q"] - df["q_des"]
    e_qd = df["qd"] - df["qd_des"]
    e_q_rms = np.sqrt(np.mean(e_q[mask]**2))
    e_qd_rms = np.sqrt(np.mean(e_qd[mask]**2))

    # friction
    tau_res = df["tau_res"][mask_fric]
    tau_hat = df["tau_hat_f"][mask_fric]
    fric_err = tau_hat - tau_res
    fric_rms = np.sqrt(np.mean(fric_err**2))
    fric_r2 = 1.0 - np.sum(fric_err**2) / np.sum((tau_res - tau_res.mean())**2)

    # control effort
    tau = df["tau_cmd"][mask]
    tau_rms = np.sqrt(np.mean(tau**2))
    tau_rough = np.mean(np.abs(np.diff(tau)))

    print(path)
    print(f"  e_q_rms   = {e_q_rms:.4f}")
    print(f"  e_qd_rms  = {e_qd_rms:.4f}")
    print(f"  fric_rms  = {fric_rms:.4f}")
    print(f"  fric_R2   = {fric_r2:.3f}")
    print(f"  tau_rms   = {tau_rms:.4f}")
    print(f"  tau_rough = {tau_rough:.4f}")

# analyze("./csvs/friction_run_joint1_2025-11-19_11-31-52.csv")
# analyze("./csvs/friction_run_joint1_2025-11-19_14-57-24.csv")
for csv_file in folder.glob("*.csv"):
    analyze(csv_file)
