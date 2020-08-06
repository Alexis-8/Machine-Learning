import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("web_traffic.tsv", delimiter="\t")
np.sum(np.isnan(data[:,1])) # 8

x = data[~np.isnan(data[:,1]),0]
y = data[~np.isnan(data[:,1]),1]

# linear approx poly = 1 :

fp1, residuals, rank, sv, rcond = np.polyfit(x,y,1,full=True)
print("Параметры модели:%s" %fp1) # Параметры модели: [  2.59619213 989.02487106]
f1 = np.poly1d(fp1) # 2.596 x + 989

# poly = 2 :
fp2 = np.polyfit(x,y,2) # [ 1.05322215e-02 -5.26545650e+00  1.97476082e+03]
f2 = np.poly1d(fp2) # 0.01053 x**2 - 5.265 x + 1975

# plot:

plt.scatter(x,y, s=10)

plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hist/hour")

plt.xticks([w*7*24 for w in range(10)], ["week %i" % w for w in range(10)])
plt.autoscale(tight=True)

fx = np.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx))
plt.legend()
print(x[-1])
plt.grid(True, color = "0.70")
plt.show()


