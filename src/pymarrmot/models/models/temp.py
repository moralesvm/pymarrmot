import matplotlib.pyplot as plt
import numpy as np
 
x = np.arange(-10, 10, 0.01)
y = x**2
 
#adding text inside the plot
a = float(2.0987543023874523490875423)
a_txt = 'a = %.2f' % round(a, 2)
plt_text = 'line 1\n'+ a_txt
plt.text(-5, 40, plt_text, fontsize = 22)
 
plt.plot(x, y, c='g')
 
plt.xlabel("X-axis", fontsize = 15)
plt.ylabel("Y-axis",fontsize = 15)
 
plt.show(block = True)