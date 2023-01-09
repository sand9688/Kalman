import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv('Accelerometer.csv')  # csv 읽어오기
df2 = df1.loc[:, ['seconds_elapsed', 'y', 'x']]  # 필요한 데이터만 추출

print(df2)

df_t = list(df2['seconds_elapsed'])  # time_data to list
print(df_t)


list_x = list(df2['x'])
list_y = list(df2['y'])


def df_integral(data):  # 수치적분, 사다리꼴적분 가속도 -> 속도 -> 거리
    result = [0]

    for i in range(len(data) - 1):
        n = result[i] + ((data[i + 1] + data[i]) * (df_t[i + 1] - df_t[i]) / 2)  # 수치적분, 사다리꼴적분
        result.append(round(n, 7))
    return result


x_2 = df_integral(list_x) # x 가속도 -> 속도
x_3 = df_integral(x_2) # x 속도 -> 거리

y_2 = df_integral(list_y) # y 가속도 - > 속도
y_3 = df_integral(y_2) # y 속도 -> 거리

df_scatter = pd.DataFrame(zip(x_3, y_3), columns=['m_y', 'm_x']) # 리스트 열단위 데이터프레임화

plt.scatter(df_scatter['m_x'], df_scatter['m_y'])
plt.show() # 플로팅

print(df_scatter)





## 정리전

# ms_x = [0]
# ms_y = [0]
# m_x = [0]
# m_y = [0]
#
# for i in range(0, (df2.shape[0] - 1)):
#     ms_i = ms_x[i] + ((df2.iat[i + 1, 2] + df2.iat[i, 2]) * (df_t[i + 1] - df_t[i]) / 2)
#     ms_x.append(round(ms_i, 7))
#
#     ms_i = ms_y[i] + ((df2.iat[i + 1, 1] + df2.iat[i, 1]) * (df_t[i + 1] - df_t[i]) / 2)
#     ms_y.append(round(ms_i, 7))
#
# print(ms_x)
# print(ms_y)
# for i in range(0, (df2.shape[0] - 1)):
#     ms_i = m_x[i] + ((ms_x[i + 1] + ms_x[i]) * (df_t[i + 1] - df_t[i]) / 2)
#     m_x.append(round(ms_i, 7))
#
#     ms_i = m_y[i] + ((ms_y[i + 1] + ms_y[i]) * (df_t[i + 1] - df_t[i]) / 2)
#     m_y.append(round(ms_i, 7))
#
# print(m_x)
# print(m_y)
#
# df_scatter = pd.DataFrame(zip(m_y, m_x), columns=['m_y', 'm_x'])
#
# plt.scatter(df_scatter['m_x'], df_scatter['m_y'])
# plt.show()
