
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scienceplots
plt.style.use('science')
import os 
from pathlib import Path

class elecstat(object):
    def __init__(self,
        turnout, margin):
        self.turnout = turnout
        self.margin = margin

    def den_plot(self,data,bins):
        """Calculates the histogram"""
        yvalue, edges = np.histogram(data, density=True, bins = bins)
        xvalue = (0.5*(edges[1:] + edges[:-1]))
        return [xvalue, yvalue]

    def rvm(self):
        """The RVM Model"""
        margin = np.array([])
        for i in self.turnout:
            w = np.random.uniform(0, 1,3)
            w = np.sort(w)
            m = (w[2] - w[1]) * i / np.sum(w)
            margin = np.append(margin,m)
        return margin

    def calcstat(self):
        """Calculate the M/<M> & Nu/<Nu> from data"""
        M_norm = self.margin / np.mean(self.margin)
        Nu = self.margin / self.turnout
        Nu_norm = Nu / np.mean(Nu)
        return M_norm, Nu_norm

    def calcstat_rvm(self):
        """Calculate the M/<M> & Nu/<Nu> from rvm"""
        M_rvm = self.rvm()
        M_rvm_norm = M_rvm / np.mean(M_rvm)
        Nu_rvm = M_rvm / self.turnout
        Nu_rvm_norm = Nu_rvm / np.mean(Nu_rvm)
        return M_rvm_norm,Nu_rvm_norm

    def plotter(self,bins):
        M_norm, Nu_norm = self.calcstat()
        M_rvm_norm,Nu_rvm_norm = self.calcstat_rvm()
        datapoints = []
        for i in [M_rvm_norm, M_norm, Nu_rvm_norm, Nu_norm]:
            xy = self.den_plot(i,bins)
            datapoints.append(xy)
        return datapoints

def plotting(filename, file, datapoints,pp):
    """Plotting"""
    # Number of rows and coloums of the plot
    ncols, nrows = 2, 1

    fig = plt.figure(figsize=(6, nrows * 3), dpi=150)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8, wspace=0.2, hspace=0.5)
    plt.suptitle(f"Election Statistics: {filename}", fontsize=15, y=0.95)

    labels = [['$M/<M>$', '$f(M/<M>)$', 'Scaled Margin'],
              [r'$\mu/<\mu>$',r'$f(\mu/<\mu>)$','Scaled Specific Margin']]
    
    for p, datapoint in enumerate(datapoints):
        ax = plt.subplot(nrows, ncols, (p+2)//2)
        if p%2 == 0:
            ax.plot(datapoint[0], datapoint[1], label = "RVM")
        else:
            ax.scatter(datapoint[0], datapoint[1], label = "Data", s=5, color="green")

        # Set logarithmic scale for better visualization
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc="upper right", fontsize=4)
        ax.set_xlabel, ax.set_ylabel, ax.set_title = labels[((p+2)//2)-1]
    
    pp.savefig(fig)




if __name__ == "__main__":
    # Get all the files in the directory
    path = os.getcwd()
    loc = new_directory = os.path.join(path, "Election Data")
    files = Path(loc).glob('*.csv')
    files = sorted(files)
    pp = PdfPages("Election Statistics.pdf")
    figures = []
    for file in files:
        filename = str(file)
        filename = filename.split("/")
        data = pd.read_csv(file, low_memory=False)
        data = data.loc[(data["Position"] == 1) & data['Margin'].notna() & (data['Valid_Votes'] != 0)]

        # Input DATA
        margin = data['Margin']
        turnout = data['Valid_Votes']
        bins = 70

        elcdata = elecstat(turnout,margin)
        datapoints = elcdata.plotter(bins)

        plotting(filename[-1], file, datapoints,pp)
        print(filename[-1])

    pp.close()
    
    
