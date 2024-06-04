import codecademylib3_seaborn
import matplotlib.pyplot as plt
from data import bs, bs_000000001, bs_01

iterations = range(100)

plt.plot(iterations, bs_01)
plt.xlabel("Iterations")
plt.ylabel("b value")
plt.show()

num_iterations = 800
convergence_b = 45

#learning rate is far too large


import codecademylib3_seaborn
import matplotlib.pyplot as plt
from data import bs, bs_000000001, bs_01

iterations = range(1400)

plt.plot(iterations, bs_000000001)
plt.xlabel("Iterations")
plt.ylabel("b value")
plt.show()

#learning rate is far too small
