import IMP
import IMP.isd
import numpy as np
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
            rest = rf.ChemicalRateRestraint(m, ks, n, sigma, c, fc, time)
        else:
            rest = rf.VoidRestraint(m)
        crr.append(rest)
        tmp.append(rest)
    rest_dict[c] = tmp

likelihood = IMP.core.RestraintsScoringFunction(crr)
priors = [IMP.isd.JeffreysRestraint(m, sigma)]
sf = IMP.core.RestraintsScoringFunction(crr + priors)
prior = IMP.core.RestraintsScoringFunction(priors)

# setting up montecarlo with movers
mc = IMP.core.MonteCarlo(m)
mc.set_scoring_function(sf)
maxstep = 0.05

mvs = []
for k in ks:
    mvs.append(
        IMP.core.NormalMover([k], IMP.FloatKeys([IMP.FloatKey("nuisance")]), maxstep)
    )

mvs.append(
    IMP.core.NormalMover([sigma], IMP.FloatKeys([IMP.FloatKey("nuisance")]), maxstep)
)


sm = IMP.core.SerialMover(mvs)
mc.add_mover(sm)
mc.set_return_best(False)
temp = 1.0
mc.set_kt(temp)
nsteps = 100000

bestscore = sf.evaluate(False)
tol = 0.5


# run simulations
for nloop in range(nsteps):
    mc.optimize(len(ks))

    # adapte the mover amplitudes
    rates = [k.get_scale() for k in ks]
    for n, rate in enumerate(rates):
        mvs[n].set_sigma(rate / 10)
    mvs[-1].set_sigma(sigma.get_scale() / 10)

    # print-out
    if sf.evaluate(False) < bestscore:

        bestscore = sf.evaluate(False)
        lk = likelihood.evaluate(False)
        pr = prior.evaluate(False)
        sumerra = 0.0
        nerra = 0

        sumerrb = 0.0
        nerrb = 0
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
        print("***", sumerra / nerra, sumerrb / nerrb, bestscore, lk, pr, temp, nloop)

