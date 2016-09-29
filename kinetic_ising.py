import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.special import erfinv
	
def bool2int(x):				#Transform bool array into positive integer
    y = 0L
    for i,j in enumerate(np.array(x)[::-1]):
#        y += j<<i
        y += long(j*2**i)
    return y
    
def bitfield(n,size):			#Transform positive integer into bit array
    x = [int(x) for x in bin(n)[2:]]
    x = [0]*(size-len(x)) + x
    return np.array(x)

def subPDF(P,rng):
	subsize=len(rng)
	Ps=np.zeros(2**subsize)
	size=int(np.log2(len(P)))
	for n in range(len(P)):
		s=bitfield(n,size)
		Ps[bool2int(s[rng])]+=P[n]
	return Ps
	
def Entropy(P):
	E=0.0
	for n in range(len(P)):
		if P[n]>0:
			E+=-P[n]*np.log(P[n])
	return E
	

def MI(Pxy, rngx, rngy):
	size=int(np.log2(len(Pxy)))
	Px=subPDF(Pxy,rngx)
	Py=subPDF(Pxy,rngy)
	I=0.0
	for n in range(len(Pxy)):
		s=bitfield(n,size)
		if Pxy[n]>0:
			I+=Pxy[n]*np.log(Pxy[n]/(Px[bool2int(s[rngx])]*Py[bool2int(s[rngy])]))
	return I
	
def TSE(P):
	size=int(np.log2(len(P)))
	C=0
	for npart in np.arange(1,0.5+size/2.0).astype(int):	
		bipartitions = list(combinations(range(size),npart))
		for bp in bipartitions:
			bp1=list(bp)
			bp2=list(set(range(size)) - set(bp))
			C+=MI(P, bp1, bp2)/float(len(bipartitions))
	return C
	
def KL(P,Q):
	D=0
	for i in range(len(P)):
		D+=P[i]*np.log(P[i]/Q[i])
	return D
    
def JSD(P,Q):
	return 0.5*(KL(P,Q)+KL(Q,P))

	
def PCA(h,J):
	size=len(h)
	P=get_PDF(h,J,size)
	m,C=observables(P,size)
	C=0.5*(C+np.transpose(C))
	w,v = np.linalg.eig(C)
	return w,v
	

class kinetic_ising:
	def __init__(self, netsize):	#Create ising model
	
		self.size=netsize
		self.h=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.Beta=1
		self.randomize_state()

	
	def randomize_state(self):
		self.s = np.random.randint(0,2,self.size)*2-1		
		
	def pdfMC(self,T):	#Get mean and correlations from Monte Carlo simulation of the kinetic ising model
		self.P=np.zeros(2**self.size)
		self.randomize_state()
		for t in range(T):
			self.GlauberStep()
			n=bool2int((self.s+1)/2)
			self.P[n]+=1.0/float(T)

	def random_wiring(self):	#Set random values for h and J
		self.h=np.random.rand(self.size)*2-1
		self.J=np.random.randn(self.size,self.size)/float(self.size)
				
	def random_rewire(self):
		if np.random.rand(1)>0.5:
			self.h[np.random.randint(self.size)]=np.random.randn(1)
		else:
			i=np.random.randint(self.size)
			j=np.random.randint(self.size)
			self.J[i,j]=np.random.randn(1)
			
	def GlauberStep(self):
		s=self.s.copy()
		for i in range(self.size):
			eDiff = self.deltaE(s,i)
			if np.random.rand(1) < 1.0/(1.0+np.exp(self.Beta*eDiff)):    # Glauber
				self.s[i] = -self.s[i]
		
	def deltaE(self,s,i=None):
		if i is None:
			return 2*(s*self.h + s*np.dot(s,self.J) )
		else:
			return 2*(s[i]*self.h[i] + np.sum(s[i]*(self.J[:,i]*s)))
		
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.C[i,j]-=self.m[i]*self.m[j]
		for i in range(self.size):
			for j in range(self.size):
				self.D[i,j]-=self.m[i]*self.m[j]
			
	def observables(self):	#Get mean and correlations from Monte Carlo simulation of the kinetic ising model
		self.m=np.zeros(self.size)
		self.D=np.zeros((self.size,self.size))
		N=float(2**self.size)
		for n in range(2**self.size):
			s=bitfield(n,self.size)*2-1
			for i in range(self.size):
#				self.m[i]+=s[i]*self.P[n]
				eDiff = self.deltaE(s,i)
				pflip = 1.0/(1.0+np.exp(eDiff))
				self.D[:,i]+=(s[i]*s*(1-pflip) - s[i]*s*pflip)*self.P[n]
				self.m[i]+=(s[i]*(1-pflip) - s[i]*pflip)*self.P[n]
		for i in range(self.size):
			for j in range(self.size):
				self.D[i,j]-=self.m[i]*self.m[j]

	def observables_sample(self,sample):	#Get mean and correlations from Monte Carlo simulation of the kinetic ising model
		self.m=np.zeros(self.size)
		self.D=np.zeros((self.size,self.size))
		ns,P=np.unique(sample,return_counts=True)
		P=P.astype(float)
		P/=np.sum(P)
		for ind,n in enumerate(ns):
			s=bitfield(n,self.size)*2-1
			eDiff = self.deltaE(s)
			pflip=1.0/(1.0+np.exp(self.Beta*eDiff))
			self.m+= (s*(1-2*pflip))*P[ind]
			d1, d2 = np.meshgrid(s*(1-2*pflip),s)
			self.D+= d1*d2*P[ind]
#			for i in range(self.size):
#				eDiff = self.deltaE(s,i)
#				pflip = 1.0/(1.0+np.exp(eDiff))
#				self.D[:,i]+=(s[i]*s*(1-2*pflip))*P[ind]
#				self.m[i]+=(s[i]*(1-2*pflip))*P[ind]
		for i in range(self.size):
			for j in range(self.size):
				self.D[i,j]-=self.m[i]*self.m[j]
				

	def generate_sample(self,T):	#Generate a series of Monte Carlo samples
		self.randomize_state()
		samples=[]
		for t in range(T):
			self.GlauberStep()
			n=bool2int((self.s+1)/2)
			samples+=[n]
		return samples
		
		
	def independent_model(self, m):		#Set h to match an independen models with means m
		self.h=np.zeros((self.size))
		for i in range(self.size):
			self.h[i]=-0.5*np.log((1-m[i])/(1+m[i]))
		self.J=np.zeros((self.size,self.size))
		
	def inverse(self,m1,D1, error,sample):
		u=0.1
		count=0
		self.observables_sample(sample)
		fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.D-D1)))
		
		while fit>error:
			self.observables_sample(sample)
			dh=u*(m1-self.m)
			self.h+=dh
			dJ=u*(D1-self.D)
			self.J+=dJ
			fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.D-D1)))
			if count%10==0:
				print self.size,count,fit
			count+=1
		return fit
		
		
#	def observables_energyMClocal(self,T):
#	
#		
#		dh=np.zeros((self.size))
#		dJ=np.zeros((self.size,self.size))
#	
#		self.HCg=0
#		self.HC=np.zeros(self.size)
#		E2=np.zeros(self.size)
#		
#		self.randomize_state()
#		for t in range(T):
#			sp=self.s.copy()
#			self.GlauberStep()
#			
#			H=self.h + np.dot(sp,self.J)
#			F=self.s-np.tanh(H)
#			G=F-H/np.cosh(H)**2
#			
#			E2+=(H*(self.s-np.tanh(self.Beta*H)))**2/float(T)
#			for i in range(self.size):
#				dh[i] += (2*H[i]*F[i]*G[i] + H[i]**2*F[i]**3)/float(T)
#				for j in range(self.size):
##					if not i==j:
#						dJ[j,i] += (2*sp[j]*H[i]*F[i]*G[i] + sp[j]*H[i]**2*F[i]**3)/float(T)

#		self.HC+=self.Beta**2*E2
#		self.HCg+=self.Beta**2*np.sum(E2)


#		return dh,dJ


#		
#	def observables_energyMClocal(self,T):
#	
#		
#		dh=np.zeros((self.size))
#		dJ=np.zeros((self.size,self.size))
#	
#		self.HC=0
##		self.HC=np.zeros(self.size)
##		C=0
#		self.randomize_state()
#		for t in range(T):
#			sp=self.s.copy()
#			self.GlauberStep()
#			if t%(T/10)==0:
#				self.randomize_state()
#			
#			H=self.h + np.dot(sp,self.J)
#			dP=self.s-np.tanh(H)
#			E=-sum(H*(self.s-np.tanh(self.Beta*H))) 
#			dE=-(dP-H/np.cosh(H)**2)
#			
#			self.HC+=(self.Beta**2)*(E**2)/float(T)
#			for i in range(self.size):
#				dh[i] += (2*E*dE[i] + E**2*dP[i])/float(T)
#				for j in range(self.size):
##					if not i==j:
#						dJ[j,i] += sp[j]*(2*E*dE[i] + E**2*dP[i])/float(T)

#		return dh,dJ
		
		
	def HeatCapacityMC(self,T):
	
		
		dh=np.zeros((self.size))
		dJ=np.zeros((self.size,self.size))
	
		self.HCg=0
		self.HC=np.zeros(self.size)
#		C=0
		self.randomize_state()
		for t in range(T):
			sp=self.s.copy()
			self.GlauberStep()
			if t%(T/10)==0:
				self.randomize_state()
			
			H=self.h + np.dot(sp,self.J)
			dS=self.Beta**3*(H/np.cosh(self.Beta*H))**2
			F=2*(H-H**2*np.tanh(H))/np.cosh(H)**2
			
			self.HC+=dS/float(T)
			for i in range(self.size):
				dh[i] += F[i]/float(T)
				for j in range(self.size):
#					if not i==j:
						dJ[j,i] += sp[j]*F[i]/float(T)
						
		self.HCg+=np.sum(self.HC)

		return dh,dJ
