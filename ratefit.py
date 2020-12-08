import IMP
import IMP.isd
import math
import numpy
import random

import time, sys

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format(
        "#" * block + "-" * (barLength - block), progress * 100, status
    )
    sys.stdout.write(text)
    sys.stdout.flush()


class ChemicalEquations(object):
    """
    initial integrator to get the experimental data points
    """

    def __init__(self, initial_conc_reagent, ks, initial_conc_reactants):
        self.ks = ks
        self.reagent = initial_conc_reagent
        self.reactants = initial_conc_reactants.copy()

    def do_integrate(self, time, nintegration):
        dt = time / nintegration
        for i in numpy.linspace(0, time, nintegration):
            dreagent = -sum(self.ks) * self.reagent * dt
            dreactants = []
            for n, k in enumerate(self.ks):
                dreactants.append(k * self.reagent * dt)
            self.reagent = self.reagent + dreagent
            for n, k in enumerate(self.ks):
                self.reactants[n] = self.reactants[n] + dreactants[n]
            # print(self.reagent,self.reactants)
        return self.reactants


def setupnuisance(m, initialvalue, minvalue, maxvalue, isoptimized=True):
    """
    function to setup the free baysian parameters
    """
    nuisance = IMP.isd.Scale.setup_particle(IMP.Particle(m), initialvalue)
    if minvalue:
        nuisance.set_lower(minvalue)
    if maxvalue:
        nuisance.set_upper(maxvalue)
    nuisance.set_is_optimized(nuisance.get_nuisance_key(), isoptimized)
    return nuisance


class TruncatedNormal(object):
    """
    class that encodes the pdf of truncated normal distribution
    """

    def __init__(self, a, b, mean):
        self.a = a
        self.b = b
        self.mean = mean

    def get_normal_pdf(self, sigma, x):
        return (
            1.0
            / math.sqrt(2.0 * math.pi)
            * math.exp(-0.5 * ((x - self.mean) / sigma) ** 2)
        )

    def phi(self, x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def get_normalization(self, sigma):
        return sigma * (
            self.phi((self.b - self.mean) / sigma)
            - self.phi((self.a - self.mean) / sigma)
        )

    def get_pdf(self, sigma, x):
        return self.get_normal_pdf(sigma, x) / self.get_normalization(sigma)

    def get_log_pdf(self, sigma, x):
        a = 0.5 * ((x - self.mean) / sigma) ** 2
        b = math.log(sigma)
        c = math.log(
            self.phi((self.b - self.mean) / sigma)
            - self.phi((self.a - self.mean) / sigma)
        )
        return a + b + c

    def test_pdf(self):
        mean = 0.1
        sigma = 0.1
        a = 0.0
        b = 1.0
        npoints = 10000
        cumu = 0.0
        for n, x in enumerate(numpy.linspace(a, b, npoints)):
            pdf = self.get_pdf(sigma, x)
            if n > 0:
                cumu += pdf * (b - a) / npoints
            # print(x,pdf,cumu)


class MarginalTruncatedNormal(object):
    """
    Numerical marginalization of a Truncated Normal likelihood
    """

    def __init__(self, TruncatedNormal, sigma, nintegration_points=200, nvalues=200):
        self.tn = TruncatedNormal
        a = self.tn.a
        b = self.tn.b
        smin = sigma.get_lower()
        smax = sigma.get_upper()
        sincr = (smax - smin) / nintegration_points
        self.table = {}

        for x in numpy.linspace(a, b, nvalues):
            acc = 0.0
            for s in numpy.linspace(smin, smax, nintegration_points):

                acc += self.tn.get_pdf(s, x) / s * sincr
            self.table[x] = acc

    def get_pdf(self, x):
        return self.table.get(
            x, self.table[min(self.table.keys(), key=lambda k: abs(k - x))]
        )


class ChemicalRateRestraint(IMP.Restraint):
    import math

    def __init__(
        self,
        m,
        kparticles,
        particleindex,
        sigma,
        initial_conc_reagent,
        final_conc_reactant,
        # time,
    ):
        """
        input
        """
        IMP.Restraint.__init__(self, m, "ChemicalRateRestraint %1%")
        self.kparticles = kparticles
        self.particleindex = particleindex
        self.initial_conc_reagent = initial_conc_reagent
        self.final_conc_reactant = final_conc_reactant
        self.sigma = sigma
        # self.time = time

        self.particle_list = self.kparticles + [self.sigma]
        self.tn = TruncatedNormal(
            0, self.initial_conc_reagent, self.final_conc_reactant
        )

    def unprotected_evaluate(self, da):
        ksum = sum([k.get_scale() for k in self.kparticles])
        ki = self.kparticles[self.particleindex].get_scale()
        forward_model = ki/ksum * self.initial_conc_reagent
        # forward_model=ki/ksum*self.initial_conc_reagent*(1.0-math.exp(-self.time*ksum))
        # prob=self.tn.get_log_pdf(self.sigma.get_scale(),forward_model)
        return self.tn.get_log_pdf(self.sigma.get_scale(), forward_model)

    def do_get_inputs(self):
        return [p.get_particle() for p in self.particle_list]


class VoidRestraint(IMP.Restraint):
    def __init__(self, m):
        """
        input
        """
        IMP.Restraint.__init__(self, m, "VoidRestraint %1%")

    def unprotected_evaluate(self, da):
        return 0.0

    def do_get_inputs(self):
        return []


class MarginalChemicalRateRestraint(IMP.Restraint):
    import math

    def __init__(
        self,
        m,
        kparticles,
        particleindex,
        sigma,
        initial_conc_reagent,
        final_conc_reactant,
        # time,
    ):
        """
        input
        """
        IMP.Restraint.__init__(self, m, "MarginalChemicalRateRestraint %1%")
        self.kparticles = kparticles
        self.particleindex = particleindex
        self.initial_conc_reagent = initial_conc_reagent
        self.final_conc_reactant = final_conc_reactant
        self.sigma = sigma
        # self.time = time

        self.particle_list = self.kparticles + [self.sigma]
        self.tn = TruncatedNormal(
            0, self.initial_conc_reagent, self.final_conc_reactant
        )
        self.mtn = MarginalTruncatedNormal(self.tn, self.sigma)
        print("Done")

    def unprotected_evaluate(self, da):
        ksum = sum([k.get_scale() for k in self.kparticles])
        ki = self.kparticles[self.particleindex].get_scale()
        forward_model = ki/ksum * self.initial_conc_reagent
        # forward_model = (
        #     ki / ksum * self.initial_conc_reagent * (1.0 - math.exp(-self.time * ksum))
        # )
        prob = self.mtn.get_pdf(forward_model)
        return -self.math.log(prob)

    def do_get_inputs(self):
        return [p.get_particle() for p in self.particle_list]


class ConcentrationSumPrior(IMP.Restraint):
    import math

    def __init__(self, m, kparticles, tolerance, initial_conc_reagent, time):
        IMP.Restraint.__init__(self, m, "ConcentrationSumPrior %1%")
        self.kparticles = kparticles
        self.initial_conc_reagent = initial_conc_reagent
        self.tolerance = tolerance
        self.time = time
        self.particle_list = self.kparticles

    def unprotected_evaluate(self, da):
        ksum = sum([k.get_scale() for k in self.kparticles])
        conc_sum = 0.0
        for kp in self.kparticles:
            ki = kp.get_scale()
            conc_sum += (
                ki
                / ksum
                * self.initial_conc_reagent
                * (1.0 - math.exp(-self.time * ksum))
            )
        invtol2 = 1.0 / self.tolerance / self.tolerance
        prob = (
            1.0
            / self.tolerance
            * math.exp(-0.5 * invtol2 * (conc_sum - self.initial_conc_reagent) ** 2)
        )
        return -self.math.log(prob)

    def do_get_inputs(self):
        return [p.get_particle() for p in self.particle_list]


def randomize_reacts(react_dict, delta=0.1, sparseness=0.3):
    randomized = {}
    for c in react_dict:
        tmp = []
        for r in react_dict[c]:
            if random.random() < sparseness:
                r = None
            else:
                r = r + random.uniform(-delta * r, delta * r)
            tmp.append(r)
        randomized[c] = tmp
    return randomized


def test_rates(ks, i, initial_conc_reagent, time):
    ki = ks[i]
    ksum = sum(ks)
    return ki / ksum * initial_conc_reagent * (1.0 - math.exp(-time * ksum))


def iterative_optimization(rates, sigma, scoring_function):
    ks = rates
    sf = scoring_function
    bestscore = sf.evaluate(False)
    npoints = 200
    nparticles = len(ks) + 1
    incr = 0
    for i, k in enumerate(ks):
        bestkv = k.get_scale()
        for kv in numpy.logspace(-3.0, +3.0, npoints, endpoint=True):
            update_progress(float(incr) / npoints / nparticles)
            incr += 1
            k.set_scale(kv)
            score = sf.evaluate(False)
            if score < bestscore:
                bestscore = score
                bestkv = kv
        k.set_scale(bestkv)

    bestsigma = sigma.get_scale()
    for sv in numpy.logspace(-3.0, +1.0, 200, endpoint=True):
        update_progress(float(incr) / npoints / nparticles)
        incr += 1
        sigma.set_scale(sv)
        score = sf.evaluate(False)
        if score < bestscore:
            bestscore = score
            bestsigma = sv
    sigma.set_scale(bestsigma)
