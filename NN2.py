# -*- coding: utf-8 -*-
print "Minha primeira rede neural com duas camadas"
import numpy as np
np.random.seed(1)

def nonlin(x,deriv=False):
	if (deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

X=np.array([[3,0,4], [4,6,1], [7,2,1] , [6,1,4]])
Y=np.array([[1],[0],[1],[0]])
#os outputs são sempre unitários! isso porque ele é calculado atraves da sigmoide!


W0 = 2*np.random.random((3,4)) - 1
W1 = 2*np.random.random((4,1)) - 1

for iter in xrange(600):
	l0 = X 
	l1 = nonlin(np.dot(l0,W0))
	l2 = nonlin(np.dot(l1,W1))
	#cálculo do erro
	l2erro = Y - l2

	#if (j% 10000) == 0:
	#	print "Error:" + str(np.mean(np.abs(l2erro)))

	l2delta = l2erro * nonlin(l2,True)

  	#não entendi isso
	l1erro = l2delta.dot(W1.T)
	l1delta = l1erro * nonlin(l1,True)

	W1 += l1.T.dot(l2delta)
	W0 += l0.T.dot(l1delta)



print "l2 final"
print l2
print "output esperado"
print Y


