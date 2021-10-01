"""This file contains two classes object
First class is the Div class that generates an observation data from population with one subpopulation
and uses resample methods to estimate diversity and heterozygosity values
The second class is the Fst class that generates and observation from population with two subpopulation
and uses resample methods to estimate the fst values
"""
import allel
import numpy as np
import simulation as sim


def jk_split(n_block, to_split, seed):
    """split sites/samples into blocks with unequal sizes
    n_block -- number of blocks
    to_split -- numpy array to split (self.sites_index or self.samples_index)
    """
    np.random.seed(seed)
    random = np.random.multinomial(
        n=len(to_split) - n_block,
        pvals=np.ones(n_block) / n_block,
    ) + 1

    # index of sites in each block
    indices = np.split(to_split, np.cumsum(random))[0: n_block]
    # number of sites in each fold
    sizes = list(map(len, indices))
    return indices, np.array(sizes)


def generate_samples_index(individuals):
    """return samples_index for the input ind_index
    because of diploid, individual 1 has sample index (2, 3)
    @individuals: a list of individuals index
    """
    res = np.zeros(len(individuals) * 2, dtype=int)
    for i, index in enumerate(individuals):
        res[i * 2] = index * 2
        res[i * 2 + 1] = index * 2 + 1

    return res


class Div:
    """observation of one population: for estimating the diversity and heterozygosity
    """

    def __init__(self, pop_ts, num_ind, max_sites, seed=None):
        """
        pop_ts -- population tree sequence
        num_ind -- number of observed individual
        max_sites -- number of observed sites
        seed -- default None. set seed for numpy random
        """
        self.seed = seed
        self.ts = sim.observe(pop_ts, num_ind, max_sites, num_pop=1, seed=self.seed)
        # number of individuals
        self.num_ind = num_ind
        self.num_samples = self.ts.num_samples
        self.num_sites = self.ts.num_sites
        # sample node index (num_sample_node = num_ind * 2)
        self.sample_nodes = self.ts.samples(population=0)
        # sites index
        self.sites = np.arange(self.num_sites)
        # genotype matrix (num_sites, num_sample_nodes)
        self.geno = self.ts.genotype_matrix()

        self.__site_diversity = self.ts.diversity(span_normalise=False, windows='sites')

        h2 = ((self.geno[:, ::2] + self.geno[:, 1::2]) == 1)
        self.__ind_nhets = h2.sum(0)  # number of heterozygotes sites at each ind
        self.__site_nhets = h2.sum(1)  # number of heterozygotes ind at each site

        # calculate statistics values
        self.div = np.mean(self.__site_diversity)
        self.hetero = np.mean(self.__site_nhets) / self.num_ind

    def get_site_diversity(self, sample_nodes=None):
        """returns average pairwise diversity of a set of samples across a set of sites.
        Used for resampling over samples. This function is general in the sense that samples may have duplicates.
        Sites_index is the same as the self
        sample_nodes -- the resampled sample nodes (may contain duplicate)
        """
        if sample_nodes is None:
            sample_nodes = self.sample_nodes
        num_samples = len(sample_nodes)

        # number of 1s for each site
        ac = self.geno[:, sample_nodes].sum(1)
        num_pairs = int(num_samples * (num_samples - 1) / 2)

        # number of 0s for each site
        n_different_pairs = ac * (num_samples - ac)
        return np.mean(n_different_pairs / num_pairs)

    def bootstrap_sites_diversity(self, num_boot=500):
        """bootstrap resampling over sites for site diversity
        num_boot -- num of bootstrap times (default 500)
        """
        np.random.seed(self.seed)
        weights = np.random.multinomial(
            n=self.num_sites,
            pvals=np.ones(self.num_sites) / self.num_sites,
            size=num_boot
        )
        return np.mean(self.__site_diversity * weights, axis=1)

    def bootstrap_sites_hetero(self, num_boot=500):
        """bootstrap resampling over sites for heterozygosity
        num_boot -- num of bootstrap times (default 500)
        """
        np.random.seed(self.seed + 1)
        weights = np.random.multinomial(
            n=self.num_sites,
            pvals=np.ones(self.num_sites) / self.num_sites,
            size=num_boot
        )
        return np.mean(self.__site_nhets * weights, axis=1) / self.num_ind

    def bootstrap_ind_diversity(self, num_boot=500):
        """bootstrap over individuals for diversity
        num_boot -- num of bootstrap times (default 500)
        """
        np.random.seed(self.seed)
        seeds = np.random.randint(0, 2 ** 32 - 1, num_boot)
        inputs = [sim.sample_individuals(self.sample_nodes, self.num_ind, replace=True, seed=seeds[i])
                  for i in range(num_boot)]
        values = list(map(self.get_site_diversity, inputs))
        return values

    def bootstrap_ind_hetero(self, num_boot=500):
        """bootstrap resampling over individuals for heterozygosity
        num_boot -- num of bootstrap times (default 500)
        """
        np.random.seed(self.seed)
        weights = np.random.multinomial(
            n=self.num_ind,
            pvals=np.ones(self.num_ind) / self.num_ind,
            size=num_boot
        )
        return np.mean(self.__ind_nhets * weights / self.num_sites, axis=1)

    def jackknife_one_sites_diversity(self):
        """delete one jackknife resample over sites for diversity
        """
        div_sum = self.__site_diversity.sum()
        return (div_sum - self.__site_diversity) / (self.num_sites - 1)

    def jackknife_one_sites_hetero(self):
        """delete one jackknife resample over sites for heterozygosity
        """
        div_sum = self.__site_nhets.sum()
        values = (div_sum - self.__site_nhets) / (self.num_sites - 1)
        return values / self.num_ind

    def jackknife_one_ind_diversity(self):
        """delete one jackknife resample over individuals for diversity
        """
        inputs = [np.delete(self.sample_nodes, [i * 2, i * 2 + 1]) for i in range(self.num_ind)]
        values = list(map(self.get_site_diversity, inputs))
        return values

    def jackknife_one_ind_hetero(self):
        """delete one jackknife resample over individuals for heterozygosity
        """
        div_sum = self.__ind_nhets.sum()
        values = (div_sum - self.__ind_nhets) / self.num_sites
        return values / (self.num_ind - 1)

    def jackknife_mj_sites_diversity(self, n_block):
        """delete_mj jackknife resampling methods over sits with unequal sizes for diversity
        """
        div_sum = self.__site_diversity.sum()
        indices, sizes = jk_split(n_block, self.sites, seed=self.seed)
        values = np.array([self.__site_diversity[i].sum() for i in indices])
        return (div_sum - values) / (self.num_sites - sizes), sizes

    def jackknife_mj_sites_hetero(self, n_block):
        """delete_mj jackknife resampling methods over sits with unequal sizes for heterozygosity
        """
        div_sum = self.__site_nhets.sum()
        indices, sizes = jk_split(n_block, self.sites, seed=self.seed + 1)
        values = np.array([self.__site_nhets[i].sum() for i in indices])
        return ((div_sum - values) / (self.num_sites - sizes)) / self.num_ind, sizes


# TODO: check Fst Class
class Fst:
    """observation of two population: for estimating fst
    """

    def __init__(self, pop_ts, num_ind, max_sites, seed=None):
        """
        @pop_ts: population tree sequence
        @num_ind: number of observed individual
        @max_sites: number of observed sites
        @seed: default None. set seed for numpy random
        """
        self.seed = seed
        self.ts = sim.observe(pop_ts, num_ind, max_sites, num_pop=2, seed=self.seed)
        # haploid index for popA
        self.popA_haploid = self.ts.samples(population=0)
        self.popB_haploid = self.ts.samples(population=1)
        self.sites_index = np.arange(self.ts.num_sites)
        self.ga = allel.GenotypeArray(self.ts.genotype_matrix().reshape(self.ts.num_sites, self.ts.num_samples, 1),
                                      dtype='i1')
        self.num, self.den = allel.hudson_fst(self.ga[:, self.popA_haploid, :].count_alleles(),
                                              self.ga[:, self.popB_haploid, :].count_alleles())
        # number of individuals in each population
        self.popA_num_ind = self.popB_num_ind = num_ind
        self.fst = np.sum(self.num) / np.sum(self.den)

    def get_fst_general(self, params=None):
        """returns Hudson's Fst: used for resampling over samples

        This function is general in the sense that all of:
          (popA_samples, popB_samples) may have duplicates.
        @params: wrapped for multiprocess
            @popA_samples = the samples from the first population to be used
            @popB_samples = the samples from the second population to be used
        """
        # count alleles within each population at the selected sites and ind
        if params is None:
            popA_samples, popB_samples = self.popA_haploid, self.popB_haploid
        else:
            popA_samples, popB_samples = params
        ac1 = self.ga[self.sites_index][:, popA_samples, :].count_alleles()
        ac2 = self.ga[self.sites_index][:, popB_samples, :].count_alleles()
        # calculate Hudson's Fst (weighted)
        num, den = allel.hudson_fst(ac1, ac2)
        return np.sum(num) / np.sum(den)

    def bootstrap_ind_fst(self, num_boot=500):
        """Calculate Fst while bootstrap resampling over individuals.
        uses multiprocessing.Pool to run across multiple cores.
        @num_boot = number of bootstrap times
        """
        np.random.seed(self.seed)
        seeds = np.random.randint(0, 2 ** 32 - 1, num_boot * 2)
        inputs = [(sim.sample_individuals(self.popA_haploid, self.popA_num_ind, replace=True, seed=seeds[i]),
                   sim.sample_individuals(self.popB_haploid, self.popB_num_ind, replace=True, seed=seeds[i + 1])
                   ) for i in range(0, num_boot * 2, 2)]

        return list(map(self.get_fst_general, inputs))

    def jackknife_one_ind_fst(self):
        """
        Calculate Fst while jackknife resampling over individuals.
        uses multiprocessing.Pool to run across multiple cores
        """
        inputs = [(np.delete(self.popA_haploid, [i, i + 1]), self.popB_haploid)
                  for i in range(0, len(self.popA_haploid), 2)] + \
                 [(self.popA_haploid, np.delete(self.popB_haploid, [i, i + 1]))
                  for i in range(0, len(self.popB_haploid), 2)]
        values = list(map(self.get_fst_general, inputs))
        return values

    def bootstrap_sites_fst(self, num_boot=500):
        """Calculate Fst for bootstrap of sites.
        @num_boot = number of bootstrap re-samplings
        """
        np.random.seed(self.seed)
        weights = np.random.multinomial(
            n=self.ts.num_sites,
            pvals=np.ones(self.ts.num_sites) / self.ts.num_sites,
            size=num_boot
        )
        return (self.num * weights).sum(1) / (self.den * weights).sum(1)

    def jackknife_one_sites_fst(self):
        """Calculate Fst for jackknife of sites.
        """
        num_sum, den_sum = self.num.sum(), self.den.sum()
        return (num_sum - self.num) / (den_sum - self.den)

    def jackknife_mj_sites_fst(self, n_block):
        index, sizes = jk_split(n_block, self.sites_index, self.seed)
        num = np.array([np.sum(self.num[i]) for i in index])
        den = np.array([np.sum(self.den[i]) for i in index])
        return (self.num.sum() - num) / (self.den.sum() - den), sizes
