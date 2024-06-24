
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

def rvm(turnout):
    margin = np.array([])
    for i in turnout:
        w = np.random.uniform(0, 1,3)
        w = np.sort(w)
        m = (w[2] - w[1]) * i / np.sum(w)
        margin = np.append(margin,m)
    return margin

def den_plot(data,bins):
    yvalue, edges = np.histogram(data, density=True, bins = bins)
    xvalue = (0.5*(edges[1:] + edges[:-1]))
    return xvalue, yvalue

# Read Data
data = pd.read_csv('Kerala_AE.csv')
# data = pd.read_csv('All_States_GE.csv', low_memory=False)
data = data.loc[(data["Position"] == 1) & data['Margin'].notna() & (data['Valid_Votes'] != 0)]

# Input DATA
margin = data['Margin']
turnout = data['Valid_Votes']
bins = 70


# Calculate the M/<M> & Nu/<Nu> from data
M_norm = margin / np.mean(margin)
Nu = margin / turnout
Nu_norm = Nu / np.mean(Nu)

# Calculate the M/<M> & Nu/<Nu> from rvm
M_rvm = rvm(turnout)
M_rvm_norm = M_rvm / np.mean(M_rvm)
Nu_rvm = M_rvm / turnout
Nu_rvm_norm = Nu_rvm / np.mean(Nu)

# Calculating the histogram from data
M_x, M_y = den_plot(M_norm,bins)
Nu_x, Nu_y = den_plot(Nu_norm,bins)

# Calculating the histogram from rvm
M_rvm_x, M_rvm_y = den_plot(M_rvm_norm, bins)
Nu_rvm_x, Nu_rvm_y = den_plot(Nu_rvm_norm, bins)


datapoints = [[[M_rvm_x, M_rvm_y],[M_x, M_y]],[[Nu_rvm_x, Nu_rvm_y],[Nu_x, Nu_y]]]

ncols = 2
nrows = 1

plt.figure(figsize=(6, nrows * 3), dpi=150)
plt.subplots_adjust(left=0.1,
    bottom=0.1,
    right=0.9,
    top=0.8,
    wspace=0.2,
    hspace=0.5)
# plt.suptitle(f"Election Statistics: India Genral Elections", fontsize=15, y=0.95)
plt.suptitle(f"Election Statistics: Kerala Assembly Elections", fontsize=15, y=0.95)

# Plotting
for p, datapoint in enumerate(datapoints):
    ax = plt.subplot(nrows, ncols, p+1)
    ax.plot(datapoint[0][0], datapoint[0][1], label = "RVM")
    ax.scatter(datapoint[1][0], datapoint[1][1], label = "Data", s=5)

    # Set logarithmic scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc="upper right", fontsize=4)

    # Set labels and title

    if p == 0:
        ax.set_xlabel(r'$M/<M>$')
        ax.set_ylabel(r'$f(M/<M>)$')
        ax.set_title('Scaled Margin')

    if p == 1:
        ax.set_xlabel(r'$\mu/<\mu>$')
        ax.set_ylabel(r'$f(\mu/<\mu>)$')
        ax.set_title('Scaled Specific Margin')
    

# Show plot
plt.show()
