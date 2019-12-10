import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
sns.set()


m = 100
k = 0.5 # коэффициент диффузии

eps = float(input())
# eps = 10e-6 # точность


m = 100
h = 3/m
t = (h ** 2) / (4 * k)
a = 1 - (4 * t * k / (h ** 2)) # actually this is zero
b = t * (k / (h ** 2) - 1 / (2 * h))
c = t * (k / (h ** 2) + 1 / (2 * h))
d = t * (k / (h ** 2))


f_in = open("grid.txt", 'r')
grid = []
for i in f_in.readlines():
    grid.append(i.strip().split())
grid = np.array(grid, dtype=int)
f_in.close()


u_old = np.zeros((m,m))
u_old[:,0] = 1
u_new = u_old.copy()

for _ in range(100000):
    u_new = a * u_old

    u_new[:, 1:] += c * u_old[:,:-1] * (grid[:,:-1] != 0)
    u_new[:, 1:] += b * u_old[:, 1:] * (grid[:, :-1] == 0)
    u_new[:, 0] += b * u_old[:, 0]

    u_new[:, :-1] += b * u_old[:,1:] * (grid[:,1:] != 0)
    u_new[:, :-1] += c * u_old[:, :-1] * (grid[:, 1:] == 0)
    u_new[:, -1] += c * u_old[:, -1]



    u_new[1:, :] += d * u_old[:-1,:] * (grid[:-1,:] != 0)
    u_new[1:, :] += d * u_old[1:, :] * (grid[:-1, :] == 0)
    u_new[0,: ] += d * u_old[0, :]

    u_new[:-1, :] += d * u_old[1:,:] * (grid[1:,:] != 0)
    u_new[:-1, :] += d * u_old[:-1, :] * (grid[1:, :] == 0)
    u_new[-1, :] += d * u_old[-1, :]


    u_new[grid == 0] = 0
    e = np.max(np.abs(u_new-u_old))
    u_old = u_new.copy()
    if e < eps:
        break


print(m)
for i in range(m):
    for j in range(m):
        print("%.12f" %(u_old[i,j]), end=" ")


ax = sns.heatmap(u_old, cmap="Greens")
# ax = sns.heatmap(u_old)
plt.show()
