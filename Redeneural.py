# -*- coding: utf-8 -*-
print "Minha primeira rede neural"

import numpy as np
np.random.seed(1)

#função sigmóide :)
def nonlin(x,deriv=False):
	if (deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

#X=matriz de inputs
X=np.array([[3,0,4], [4,6,1], [7,2,1] , [6,1,4]])
#print X

#Y=matriz de outputs esperados
Y=np.array([[0],[1],[1],[0]])
#print Y

#W=matriz de pesos 3x1 de -1 a +1
W0= 2*np.random.random((3,1)) - 1
#print W0

for iter in xrange(300):
	l0 = X 
	l1 = nonlin(np.dot(l0,W0))
	#cálculo do erro
	l1erro = Y-l1
	l1delta = l1erro * nonlin(l1,True)
	W0 += np.dot(l0.T,l1delta)
	#print "l1"
	#print l1
	#print "l1erro"
	#print l1erro

#print "acabaram as 8 vezes"
print "l1 final"
print l1
print "output esperado"
print Y

T=np.array([[4,6,1], [3,0,4], [7,2,1] , [6,1,4]])
PLP = nonlin(np.dot(T,W0))
print PLP





