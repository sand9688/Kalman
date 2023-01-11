import numpy as np
from numpy import transpose
from numpy.linalg import inv

#참조 : https://gaussian37.github.io/autodrive-ose-lkf_image_tracking/
# 파라미터 초기화
def init():
    global dt, A, H, Q, R, P, x

    dt = 1

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
    Q = np.eye(4)
    print(Q)

    R = np.array([
        [50, 0],
        [0, 50]
    ])

    P = 100 * np.eye(4)
    print(P)

    x = np.array([0, 0, 0, 0])


# 선형 칼만필터 함수
# input: 객체 검출한 측정 좌표 (x_, y_)
# output: 칼만 필터 적용한 객체 좌표와 속도 (추정 x좌표, x축 속도, 추정 y좌표, y축 속도)

def KalmanTracking(x_, y_):
    global A, H, Q, R, P, x

    xp = A @ x
    Pp = A @ P @ transpose(A) + Q

    K = Pp @ transpose(H) @ inv(H @ Pp @ transpose(H) + R)

    z = (x_, y_)
    x = xp + K @ (z - H @ xp)
    P = Pp - K @ H @ Pp

    return x


init()
a=KalmanTracking(1, 1)
print(a)