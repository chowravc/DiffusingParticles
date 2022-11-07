import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

def main():

	# Number of plus particles
	nPlus = 1000

	# Number of minus particles
	nMinus = 1000

	# Number of frames
	n = 10000

	# Mean drift
	mean = 0 # cm

	# Grid dimension
	dims = [10, 10] # [cm]

	# Time step
	dt = 0.1 # s

	# Diffusion coefficients for both defects
	DPlus = 0.5 # cm^2/s
	DMinus = 0.5 # cm^2/s

	# Annihilation tolerange
	tolerance = 0.05 # cm
	tolSq = tolerance**2 # cm^2

	# Mean step size
	sigPlus = np.sqrt(4*DPlus*dt) # cm
	sigMinus = np.sqrt(4*DMinus*dt) # cm

	# Preset start locations
	# xssPlus = np.array([[-10, 10]])
	# yssPlus = np.array([[-10, 10]])

	# xssMinus = np.array([[10, -10]])
	# yssMinus = np.array([[-10, 10]])

	# Random start locations
	xssPlus = np.array([np.random.uniform(-dims[0]//2, dims[0]//2, [nPlus])])
	yssPlus = np.array([np.random.uniform(-dims[1]//2, dims[1]//2, [nPlus])])

	xssMinus = np.array([np.random.uniform(-dims[0]//2, dims[0]//2, [nMinus])])
	yssMinus = np.array([np.random.uniform(-dims[1]//2, dims[1]//2, [nMinus])])

	# Keeps track of particles remaining
	nParts = [nPlus+nMinus]

	for f in range(n):

		# Take steps
		dxsPlus = np.random.normal(loc=mean, scale=sigPlus, size=(len(xssPlus[0]))) # cm
		dysPlus = np.random.normal(loc=mean, scale=sigPlus, size=(len(yssPlus[0]))) # cm

		dxsMinus = np.random.normal(loc=mean, scale=sigMinus, size=(len(xssMinus[0]))) # cm
		dysMinus = np.random.normal(loc=mean, scale=sigMinus, size=(len(yssMinus[0]))) # cm

		# Check if two particles touch
		newXsPlus = xssPlus[-1] + dxsPlus # cm
		newYsPlus = yssPlus[-1] + dysPlus # cm

		newXsMinus = xssMinus[-1] + dxsMinus # cm
		newYsMinus = yssMinus[-1] + dysMinus # cm

		flaggedPlus = []
		flaggedMinus = []

		for i in range(len(newXsPlus)):

			if not np.isnan(xssPlus[-1][i]) and i not in flaggedPlus:
				for j in range(len(newXsMinus)):

					if not np.isnan(xssMinus[-1][j]) and j not in flaggedMinus:
						
						delta = (newXsPlus[i]-newXsMinus[j])**2+(newYsPlus[i]-newYsMinus[j])**2

						if delta < tolSq:

							print(f'Plus {i} and Minus {j} annihilate on frame {f}')

							# Flag annihilated particles
							flaggedPlus.append(i)
							flaggedMinus.append(j)

							# Plot annihilation locations
							plt.scatter(newXsPlus[i], newYsPlus[i], marker='*', color='r', zorder=2)
							plt.scatter(newXsMinus[j], newYsMinus[j], marker='*', color='b', zorder=2)

							# Update new positions to NaN
							newXsPlus[i] = np.nan
							newYsPlus[i] = np.nan

							newXsMinus[j] = np.nan
							newYsMinus[j] = np.nan


		# Increment particles
		xssPlus = np.concatenate((xssPlus, np.array([newXsPlus])))
		yssPlus = np.concatenate((yssPlus, np.array([newYsPlus])))

		xssMinus = np.concatenate((xssMinus, np.array([newXsMinus])))
		yssMinus = np.concatenate((yssMinus, np.array([newYsMinus])))

		# Update remaining particles
		nParts.append(nParts[-1]-len(flaggedPlus)-len(flaggedMinus))

	# Take transpose of position arrays
	xssPlus = xssPlus.T
	yssPlus = yssPlus.T

	xssMinus = xssMinus.T
	yssMinus = yssMinus.T

	# Converting number of particles to NumPy array
	nParts = np.asarray(nParts)

	# Displaying diffusions
	for i in range(len(xssPlus)):

		plt.plot(xssPlus[i], yssPlus[i], linewidth=0.5, zorder=1)

	for i in range(len(xssMinus)):

		plt.plot(xssMinus[i], yssMinus[i], linewidth=0.5, zorder=1)

	plt.gca().set_aspect('equal', adjustable='box')

	plt.show()

	# Clear plots
	plt.clf()

	# Plotting number of particles
	plt.plot(nParts)
	plt.xscale("log")
	plt.yscale("log")
	plt.show()

	print(f'\nActual D (plus): {DPlus:.4g}')
	print(f'Actual D (minus): {DMinus:.4g}\n')

	for i in range(len(xssPlus)):

		dxs = np.diff(xssPlus[i][~np.isnan(xssPlus[i])])
		dys = np.diff(yssPlus[i][~np.isnan(yssPlus[i])])

		drSqs = dxs**2 + dys**2

		meanSq = np.mean(drSqs)

		DCalc = meanSq/(8*dt)
		print(f'Calculated D (plus) ({i+1}): {DCalc:.4g}')

	print()

	for i in range(len(xssMinus)):

		dxs = np.diff(xssMinus[i][~np.isnan(xssMinus[i])])
		dys = np.diff(yssMinus[i][~np.isnan(yssMinus[i])])

		drSqs = dxs**2 + dys**2

		meanSq = np.mean(drSqs)

		DCalc = meanSq/(8*dt)
		print(f'Calculated D (minus) ({i+1}): {DCalc:.4g}')

if __name__ == '__main__':
	main()