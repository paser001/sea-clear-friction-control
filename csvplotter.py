import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("csvs/friction_run_joint1_2025-10-27_15-56-21.csv")

t_ray= np.array(df["t"])
qd_ray = np.array(df["qd"])
taua_ray = np.array(df["tau_applied"])

plt.figure()
plt.plot(t_ray, qd_ray, label="velocity [rad/s]")
plt.plot(t_ray, taua_ray, label="applied torque [Nm]")
plt.xlabel("time [s]")
plt.legend()
plt.title("Joint torque and velocity")
plt.show()




