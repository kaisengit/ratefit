import IMP
import IMP.isd
import math
import numpy
import random
import ratefit as rf

m = IMP.Model()


# initialize random rates
nrates = 20
# total incubation time
time = 1.0
reaction_rates = [10.0 * random.random() for i in range(nrates)]
# initial concentration reactants
initial_concs_reactants = [0.0 for i in range(nrates)]
# initial concentration reagent
initial_concs = [0.2, 1.0, 5.0, 10.0]

# get the final yields
reacts = {}
for c in initial_concs:
    ce = rf.ChemicalEquations(c, reaction_rates, initial_concs_reactants)
    reacts[c] = ce.do_integrate(time, 10000)

# randomize the data
randreacts = rf.randomize_reacts(reacts)

# build nuisances
ks = []
for i in range(nrates):
    ks.append(rf.setupnuisance(m, 1.0, 0.001, 10.0, True))
sigma = rf.setupnuisance(m, 0.1, 0.001, 1.0, True)

# build scoring function
crr = []
priors = []
rest_dict = {}
for c in reacts:
    tmp = []
    for n, fc in enumerate(randreacts[c]):
        if fc is not None:
            rest = rf.ChemicalRateRestraint(m, ks, n, sigma, c, fc)
        else:
            rest = rf.VoidRestraint(m)
        crr.append(rest)
        tmp.append(rest)
    rest_dict[c] = tmp

likelihood = IMP.core.RestraintsScoringFunction(crr)
priors = [IMP.isd.JeffreysRestraint(m, sigma)]
sf = IMP.core.RestraintsScoringFunction(crr + priors)
prior = IMP.core.RestraintsScoringFunction(priors)

nsteps = 100000
bestscore = sf.evaluate(False)


# run optimization
for nloop in range(nsteps):
    rf.iterative_optimization(ks, sigma, sf)

    lk = likelihood.evaluate(False)
    pr = prior.evaluate(False)
    sumerra = 0.0
    nerra = 0

    sumerrb = 0.0
    nerrb = 0
    rates = [k.get_scale() for k in ks]
    for n in range(len(rates)):
        for c in initial_concs:
            tr = rf.test_rates(rates, n, c, time)
            erra = abs(tr - reacts[c][n]) / tr

            if randreacts[c][n] is not None:
                errb = abs(tr - randreacts[c][n]) / tr
                sumerrb += errb
                nerrb += 1
            else:
                errb = 0.0

            print(
                tr,
                reacts[c][n],
                erra,
                errb,
                rest_dict[c][n].unprotected_evaluate(False),
            )
            sumerra += erra
            nerra += 1

    print("***", sumerra / nerra, sumerrb / nerrb, bestscore, lk, pr, nloop)

