import numpy as np
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
from scipy.stats import chi2_contingency, norm, chi2, ttest_ind, bernoulli, pearsonr
from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import mean_squared_error

mean_effect_size = 0.05
var_effect_size = 0.03
es = np.random.normal(mean_effect_size, var_effect_size, 1000)
plt.hist(es, bins=20)
plt.show()	

alpha = 0.1
power = 0.8
x,y=[],[]
est_fdrs, actual_fdrs = [],[]
est_fgrs, actual_fgrs = [], []

powers = []
fcrs = []

for e in range(100):
	p_effect = np.random.beta(10, 90, 1)[0]
	conc = 0.
	k = 10000
	fc, fi, tc, ti = 0, 0, 0, 0
	fg, fr, tg, tr = 0, 0, 0, 0

	for i in range(k):
		aa = True
		red, green = False, False
		if random.random()<p_effect:
			es = np.random.normal(mean_effect_size, var_effect_size, 1)[0]
			aa = False
		else:
			es = 0

		if es > 0:
			green = True
			red = False
		if es < 0:
			green = False
			red = True

		a = 12
		base_conversion = np.random.beta(a, 100-a, 1)[0]
		variant_conversion = base_conversion*(1+es)

		es = (variant_conversion-base_conversion)/np.sqrt((base_conversion*(1-base_conversion) + variant_conversion*(1-variant_conversion))/2)
		es2 = sm.stats.proportion_effectsize(base_conversion*(1+np.random.normal(mean_effect_size, var_effect_size, 1)[0]), base_conversion)
		es2 = np.abs(es2)
		if es != 0.:
			n = TTestIndPower().solve_power(es2, power=0.8, ratio=1, alpha=alpha, alternative='two-sided')

		else:
			n = 0
		
		n = int(n)
		n = max(10000, min(n, 100000000))
		total_bookers_A = np.random.binomial(n, base_conversion)
		total_bookers_B = np.random.binomial(n, variant_conversion)

		no_bookers_A = n-total_bookers_A
		no_bookers_B = n-total_bookers_B

		obs = [[total_bookers_A, no_bookers_A], [total_bookers_B, no_bookers_B]]
		g, p, dof, expected = chi2_contingency(obs, lambda_="log-likelihood")


		conclusive = False
		if p<alpha:
			conclusive = True
			conc+=1.

		if aa and conclusive:
			fc += 1.
		if aa and not conclusive:
			ti += 1.
		if not aa and conclusive:
			tc += 1.
		if not aa and not conclusive:
			fi += 1.
		
		if conclusive:
			if total_bookers_B>total_bookers_A:	
				if green:
					tg += 1.
				else:
					fg += 1.
			
			if total_bookers_B<total_bookers_A:	
				if red:
					tr += 1.
				else:
					fr += 1.
				

	cr = conc/k
	tcr = (tc+1e-9)/(tc+fi+1e-9)
	fcr = fc/(fc+ti)
	fdr = fc/conc
	aar = (fc+ti)/k
	est_aar = (cr-power)/(alpha-power)
	est_abr = 1-est_aar
	

	

	est_fdr = k*((cr-power)/(alpha-power)*alpha)/conc
	est_fdrs.append(est_fdr)
	actual_fdrs.append(fdr)
	

	fgr = fg / (tg + fg)
	gr = (fg+tg)/k	
	est_fgr = k*((gr-0.88)/(alpha/2-0.88)*alpha/2)/(fg+tg)
	est_fgrs.append(est_fgr)
	actual_fgrs.append(fgr)

	print fgr, est_fgr
	#print e, 'FDR', fdr, 'EST', est_fdr, fcr, tcr
	
	#print 'Number of AAs: Actual', fc+ti, 'Estimated', k*(cr-power)/(alpha-power)
	x.append(fc+ti)
	y.append(k*(cr-power)/(alpha-power))
	powers.append(tcr)
	fcrs.append(fcr)


plt.hist(fcrs)
plt.show()
plt.hist(powers)
plt.show()


print np.sqrt(mean_squared_error(x, y)), pearsonr(x,y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.scatter(x,y)
plt.plot(x, p(x))
plt.show()

x = actual_fdrs
y = est_fdrs
print np.sqrt(mean_squared_error(x, y)), pearsonr(x,y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.scatter(x,y)
plt.plot(x, p(x))
plt.show()


x = actual_fgrs
y = est_fgrs
print np.sqrt(mean_squared_error(x, y)), pearsonr(x,y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.scatter(x,y)
plt.plot(x, p(x))
plt.show()


'''
5k
34.8651366712084 (0.9795319293628634, 4.87920249545977e-70)
0.03729674802954036 (0.9530145704169394, 1.2486694204840127e-52)
'''
 

