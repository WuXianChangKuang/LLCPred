import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = "fig_cycle_cpuuse.csv"
df = pd.read_csv(filename)
x = np.array(df.iloc[:,1].values)
y = np.array(df.iloc[:,0].values)
 
plt.figure('CYCLES_USE%')
 
plt.scatter(x,y, c = "red", marker='^')

plt.xlabel("CPU Utilization(%)", size = 10)
plt.ylabel("CPU Cycles", size = 10)

plt.savefig('./CYCLES_USE%')
plt.show()  # 显示绘图

