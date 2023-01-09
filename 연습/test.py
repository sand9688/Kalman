import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('Accelerometer.csv')  # csv 읽어오기
time_list = df1.loc[:,'seconds_elapsed']
dt = time_list.values.tolist()
x = df1.loc[:,'x'].tolist()
y = df1.loc[:,'y'].tolist()

print(x)
print(dt)


def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm for One Variable."""
    # (1) Prediction.
    x_pred = A * x_esti
    P_pred = A * P * A + Q

    # (2) Kalman Gain.
    K = P_pred * H / (H * P_pred * H + R)

    # (3) Estimation.
    x_esti = x_pred + K * (z_meas - H * x_pred)

    # (4) Error Covariance.
    P = P_pred - K * H * P_pred

    return x_esti, P

dt = dt[1]-dt[0]
n_sample = len(time_list)
x_meas_save = np.zeros(n_sample)
x_esti_save = np.zeros(n_sample)

# Initialization for system model.
A = 1
H = 1
Q = 0
R = 4
# Initialization for estimation.
x_0 = 12  # 14 for book.
P_0 = 6

x_esti, P = None, None
for i in range(n_sample):
    z_meas = x[i]
    if i == 0:
        x_esti, P = x_0, P_0
        print('z_meas :', z_meas)
        print('x_esti :', x_esti)
        print('p :', P)
    else:
        x_esti, P = kalman_filter(z_meas, x_esti, P)
        print('z_meas :', z_meas)
        print('x_esti :', x_esti)
        print('p :', P)

    x_meas_save[i] = z_meas
    x_esti_save[i] = x_esti

plt.plot(time_list, x_meas_save, 'r*--', label='Measurements')
plt.plot(time_list, x_esti_save, 'bo-', label='Kalman Filter')
plt.legend(loc='upper left')
plt.xlabel('Time ')
plt.ylabel('x')
plt.show()