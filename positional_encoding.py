
import numpy as np

mp=1000
d=4
def pe(mp,d):
	PE=np.zeros((mp,d))
	for pos in range(mp):
		for i in range(d//2):
			denominator = 10000 ** (2*i)/4
			PE[pos,2*i]=np.sin(pos/denominator)
			PE[pos,2*i+1]=np.cos(pos/denominator)
		print(PE)
PE=pe(mp,d)
tokens = [ "I","Love","AI"]
emb=np.array([
	[1,2,3,4],
	[5,6,7,8],
	[9,10,11,12]
],dtype=float)
print(emb)
sent=PE[:len(tokens)]
final_input=(emb+ sent)
print(final_input)