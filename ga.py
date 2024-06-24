# GA.py
# Dr. Matthew Smith, Swinburne University of Technology
# matthewrsmith@swin.edu.au

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def ComputeFitness(X):
    # Compute the fitness based on genetic data X
    # X is an array of dimension (NO_VAR)
    # The goal of the GA is the find the value of X which results
    # in a maximum or mimimum value of this fitness function.

    # Compute the Goldstein-Price function (two variables)
    fitness = (1.0 + (X[0]+X[1]+1.0)*(X[0]+X[1]+1.0)*(19.0-14.0*X[0]+3.0*X[0]*X[0]-14.0*X[1]+6.0*X[0]*X[1]+3.0*X[1]*X[1]))* \
        (30.0 + (2.0*X[0]-3.0*X[1])*(2.0*X[0]-3.0*X[1])*(18.0-32.0*X[0]+12.0*X[0]*X[0]+48.0*X[1]-36.0*X[0]*X[1]+27.0*X[1]*X[1]));

    return fitness

def ComputeBestKid(FITNESS):
    # Computes the fittest child
    # FITNESS is an array of dimension (NO_KIDS)
    NO_KIDS = len(FITNESS)
    BestFitness = 10000.0
    BestIndex = -1
    for i in range(NO_KIDS):
        if (np.absolute(FITNESS[i]) < BestFitness):
            BestFitness = FITNESS[i]
            BestIndex = i

    return BestIndex


def MaxMod(X):
    if (X > 0.5):
        return X
    else:
        return (1.0-X)


def ComputeNextGeneration(DNA, FITNESS, BESTINDEX, FR, SIGMA):

    NO_KIDS,NO_VAR = DNA.shape
    trial_dna = np.empty(NO_VAR)

    # Assume the next generation will be identical
    new_dna = DNA
    new_fitness = FITNESS

    # Compute a replacement for each member of this generation
    for i in range(NO_KIDS):
        # Pick parents
        Parent_A = BESTINDEX
        Parent_B = Parent_A
        Parent_C = Parent_A
        while ((Parent_A == Parent_B) or (Parent_A == Parent_C) or (Parent_B == Parent_C)):
            # Select Parent B and C randomly from the population
            Parent_B = np.random.randint(NO_KIDS)
            Parent_C = np.random.randint(NO_KIDS)

        # Compute the dna of the trial child
        for j in range(NO_VAR):
            Rf = FR*MaxMod(np.random.rand())
            Rnf = SIGMA*np.random.randn()
            trial_dna[j] = Rf*DNA[Parent_A,j]+(1.0-Rf)*DNA[Parent_B,j] + Rnf*(DNA[Parent_B,j]-DNA[Parent_C,j])

        # Compute the fitness of the trial child
        trial_fitness = ComputeFitness(trial_dna)

        if (np.abs(trial_fitness) < np.abs(FITNESS[i])):
            # This trial child has better dna. Make the change.
            new_fitness[i] = trial_fitness
            new_dna[i,:] = trial_dna

    # Return the next generations data
    return new_dna, new_fitness


# Commence the main program sequence
# Assign solution parameters
NO_KIDS = 30
NO_VAR = 2
NO_GEN = 10
FR = 1.0
SIGMA = 1.0

# Initialise the problem
kid_dna = np.random.rand(NO_KIDS, NO_VAR)
kid_fitness = np.zeros(NO_KIDS)
history_fitness = np.empty(NO_GEN)
history_dna = np.zeros((NO_GEN,NO_VAR))

# Find the child with the best dna
for i in range(NO_KIDS):
    kid_fitness[i] = ComputeFitness(kid_dna[i,:])
# Find the Index of the best kid based on fitness
BestKid = ComputeBestKid(kid_fitness)
print("The best kid in the population has fitness %g at (%g, %g)" % (kid_fitness[BestKid], kid_dna[BestKid,0], kid_dna[BestKid,1]))


# Iterate over generations to evolve the population
for gen in range(NO_GEN):
    kid_dna, kid_fitness = ComputeNextGeneration(kid_dna,kid_fitness,BestKid, FR, SIGMA)
    BestKid = ComputeBestKid(kid_fitness)
    history_fitness[gen] = kid_fitness[BestKid]
    history_dna[gen,:] = kid_dna[BestKid,:]
    # Write a report
    print("Iteration %d - best fitness = %g at (%g, %g)" % (gen, kid_fitness[BestKid], kid_dna[BestKid,0], kid_dna[BestKid,1]))


fig,ax = plt.subplots()
ax.plot(history_dna)
ax.set(xlabel='Generation Number',ylabel='DNA Values')
ax2 = ax.twinx()
ax2.semilogy(history_fitness,'r')
ax2.set(ylabel='Fitness')
plt.show()

print("====== FINAL REPORT ========")
print("Average fitness of final generation = %g" % np.mean(kid_fitness))
print("Best Kid's Fitness: %g" % kid_fitness[BestKid])
print("Best Kid's DNA Parameter values:")
for i in range(NO_VAR):
    print("-- DNA Parameter %d = %g" % (i,kid_dna[BestKid,i]))
