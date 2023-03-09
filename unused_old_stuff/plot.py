from matplotlib import pyplot as plt

with open("loss.csv", "r") as f:
    data = f.readlines()

data = [float(l.strip()) for l in data]

plt.plot(data)
plt.show()

