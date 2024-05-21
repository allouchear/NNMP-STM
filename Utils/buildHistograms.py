import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
 
 
if len(sys.argv)<2:
        print("Usage :")
        print("      give the name of csv file , number of bins and the prefix name of the output file")
        print("      Example ", sys.argv[0], " similarities.csv ", " 20 ", " hist ")
        exit(1)

#print(sys.argv)

df = pd.read_csv(sys.argv[1],sep=',')
nbins=int(sys.argv[2])
fnout=sys.argv[3]

#print(df)

nhists=len(df.columns)-1
data=[]
mu=[]
median=[]
std=[]
labels=[]
for i in range(nhists):
	data.append(df[df.columns[i]].to_numpy())
	#print(data)
	mu.append(np.mean(data[i]))
	median.append(np.median(data[i]))
	std.append(np.std(data[i]))
	labels.append(str(i))

#print(df.columns)
#print(mu)
for i in range(nhists):
	labels[i]=df.columns[i]

#print(labels)
#f, axis = plt.subplots(nhists, 1, sharex=True, figsize=(8, 6))
f, axis = plt.subplots(nhists, 1, sharex=True, gridspec_kw = {'wspace':0, 'hspace':0})

#sns.displot(x=data, kde=True)
#sns.histplot(x=data, kde=True, bins=nbins, color='b', kde_kws={'bw_method':'scott'})
st=0.20
y0=0.95
xlabel=0.02
xval=0.35
fsize=11
for i in range(nhists):
	p = sns.histplot(x=data[i], kde=True, bins=nbins, color='b', kde_kws={'bw_method':0.15}, ax=axis[i])
	#p.set(xlabel="Spectra Information Similarity", ylabel = "Frequencies",label=labels[i],yticks=[])
	#p.set(xlabel="Spectra Information Similarity", ylabel = None,label=labels[i],yticks=[])
	p.set(xlabel=" ", ylabel = None,label=labels[i],yticks=[])
	#print(p.set.__doc__)
	mutxt = "$ {} = $ {:.2f}".format(r"\bar{x}",mu[i])
	stdtxt = "$ {} = $ {:.2f}".format(r"\sigma", std[i])
	medtxt = "$ {} = $ {:.2f}".format(r"\tilde{x}", median[i])
	plt.text(xval, y0,  mutxt,fontsize=fsize,ha='left', va='top', transform=axis[i].transAxes)
	plt.text(xval, y0-st,  medtxt,fontsize=fsize,ha='left', va='top', transform=axis[i].transAxes)
	plt.text(xval, y0-2*st,  stdtxt,fontsize=fsize,ha='left', va='top', transform=axis[i].transAxes)
	plt.text(xlabel, y0,  labels[i],fontsize=fsize,ha='left', va='top', transform=axis[i].transAxes)
	#plt.xlabel("Spectra Information Similarity")
	#plt.ylabel("Frequencies")
	#axis[i].yticks([], [])

	plt.title(" ")
	#axis[i].legend()
f.text(0.08, 0.5, 'Counts', va='center', rotation='vertical')

filename=fnout+'.pdf'
plt.savefig(filename)
plt.show()
print("Voir fichier ", filename)

