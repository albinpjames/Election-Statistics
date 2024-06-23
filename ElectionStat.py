
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rvm(votes):
    c1p = np.random.uniform(0, 1)
    c2p = np.random.uniform(0, 1)
    #c3p = np.random.uniform(0, 1)

    normfact = c1p + c2p #+ c3p

    margin = (c1p*votes - c2p*votes) / normfact
    return margin




data = pd.read_csv('Kerala_AE.csv')

M_norm = data['Margin'] / np.mean(data['Margin'])
M_rvm = rvm(data['Valid_Votes'])
M_rvm_norm = M_rvm / np.mean(M_rvm)


plt.hist(M_norm, density= True,histtype='step',bins=50)
plt.hist(M_rvm_norm, density= True, bins=50)
plt.xscale('log')
plt.yscale('log')

#plt.hist(data['Valid_Votes'])
plt.show()
