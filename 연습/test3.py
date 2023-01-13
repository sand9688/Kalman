import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import transpose
from numpy.linalg import inv

df1 = pd.read_csv('Accelerometer.csv')  # csv 읽어오기
time_list = df1.loc[:,'seconds_elapsed']
time = time_list.values.tolist()
dx = df1.loc[:,'x'].tolist()
dy = df1.loc[:,'y'].tolist()

dt = time[1]-time[0]

A = np.array([
    [1, dt, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dt],
    [0, 0, 0, 1],
])

H = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

# Q와 R은 튜닝 파라미터
# Q = np.eye(4)

Q = np.diag([1.0, 1.0, 1.0, 1.0])
print(Q)

R = np.array([
    [10000, 0],
    [0, 10000]
])

P = 100 * np.eye(4)


x = np.array([0, 0, 0, 0])


# 선형 칼만필터 함수
# input: 객체 검출한 측정 좌표 (x_, y_)
# output: 칼만 필터 적용한 객체 좌표와 속도 (추정 x좌표, x축 속도, 추정 y좌표, y축 속도)

def df_integral(data):  # 수치적분, 사다리꼴적분 가속도 -> 속도 -> 거리
    result = [0]

    for i in range(len(data) - 1):
        n = result[i] + ((data[i + 1] + data[i]) * (time[i + 1] - time[i]) / 2)  # 수치적분, 사다리꼴적분
        result.append(round(n, 7))
    return result

def KalmanTracking(x_, y_):
    global A, H, Q, R, P, x

    xp = A @ x
    Pp = A @ P @ transpose(A) + Q

    K = Pp @ transpose(H) @ inv(H @ Pp @ transpose(H) + R)

    z = (x_, y_)
    x = xp + K @ (z - H @ xp)
    P = Pp - K @ H @ Pp

    return x


test_x = []
test_y = []



for i in range(len(time)):
    KalmanTracking(dx[i],dy[i])
    test_x.append(x[0])
    test_y.append(x[2])

print(test_x)
print(test_y)

x_2 = df_integral(test_x) # x 가속도 -> 속도
x_3 = df_integral(x_2) # x 속도 -> 거리

y_2 = df_integral(test_y) # y 가속도 - > 속도
y_3 = df_integral(y_2) # y 속도 -> 거리

df_scatter = pd.DataFrame(zip(x_3, y_3), columns=['m_y', 'm_x']) # 리스트 열단위 데이터프레임화

x_2 = df_integral(dx) # x 가속도 -> 속도
x_4 = df_integral(x_2) # x 속도 -> 거리

y_2 = df_integral(dy) # y 가속도 - > 속도
y_4 = df_integral(y_2) # y 속도 -> 거리

df2_scatter = pd.DataFrame(zip(x_4, y_4), columns=['m_y', 'm_x']) # 리스트 열단위 데이터프레임화


plt.subplot(2, 2, 1)
plt.plot(df_scatter['m_x'], df_scatter['m_y'])
plt.subplot(2, 2, 2)
plt.plot(df2_scatter['m_x'], df_scatter['m_y'])
plt.subplot(2, 2, 3)
plt.plot(dx, dy)
plt.subplot(2, 2, 4)
plt.plot(test_x,test_y)
plt.show()

print(df_scatter)