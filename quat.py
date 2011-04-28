from math import *
import copy

class quat:
	#builder
	def __init__(self,a=0,b=0,c=0,d=0):
		self.q=[float(a),float(b),float(c),float(d)]
	# quaternion as string or for printing
	def __str__(self):
		a=''
		lab=['i','j','k']
		a=str(self.q[0])
		if self.q[1]>=0:
			a=a+'+'
		for i in range(1,3):
				if self.q[i+1]<0:
					a=a+str(self.q[i])+lab[i-1]
				elif self.q[i+1]>=0:
					a=a+str(self.q[i])+lab[i-1]+'+'
 
		a=a+str(self.q[3])+lab[2]
		return a
	# addition of quaternions
	def __add__(self,ob2):
		s=quat()
		for i in range(4):
			s.q[i]=self.q[i]+ob2.q[i]
		return s	
	# subtraction of quaternions
	def __sub__(self,ob2):
                s=quat()
                for i in range(4):
                        s.q[i]=self.q[i]-ob2.q[i]
                return s
	# quaternion product (in R3 cross product)
	def __mul__(self,ob2):
		s=quat(0,0,0,0)
		#matrices and vector used in quaternion product (not math just for the index of the quaternion
		or1=[0,1,2,3]
		or2=[1,0,3,2]
 
		v=[[1,-1,-1,-1],[1,1,1,-1],[1,-1,1,1],[1,1,-1,1]]
		m=[or1,or2,[i for i in reversed(or2)],[i for i in reversed(or1)]]
		for k in range(4):
			a=0
			for i in range(4):
                            a=a+(self.q[m[0][i]])*(float(v[k][i])*ob2.q[m[k][i]])
			s.q[k]=a	
		return s
	# product by scalar
	def __rmul__(self,ob):
		s=quat()
		for i in range(4):
			s.q[i]=self.q[i]*ob
		return s
	# division of quaternions, quat/quat 
	def __div__(self,ob):
		s=quat()
		b=copy.deepcopy(ob)
		ob.C()
		c=(b*ob).q[0]
		s=(self*ob)
		a=(1/c)*s
		ob.C()
		return a
	# norm of a quaternion returns an scalar
	def norm(self):
		s=0
		for i in range(4):
			s=s+self.q[i]**2
		return float(s**(0.5))
	# conjugates a quaternion a: a.C() -> a=a+bi+cj+dk --> a.C()=a-bi-cj-dk
	def C(self):
		s=self
		for i in range(1,4):
			s.q[i]=s.q[i]*(-1)
		return s
	# method used to create a rotation quaternion to rotate any vector defined as a quaternion 
	# with respect to the vector vect theta 'radians' 
	def rotQuat(self,theta,vect):
		self.q=[cos(theta/2.0),vect[0]*sin(theta/2.0),vect[1]*sin(theta/2.0),vect[2]*sin(theta/2.0)]
