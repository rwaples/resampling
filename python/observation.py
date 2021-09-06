# Observation Class
import allel
import numpy as np
import simulation as sim
from multiprocessing import Pool, cpu_count


def jk_split(n_block, to_split):
    """split sites/samples into blocks with unequal sizes
    @n_block: number of blocks
    @to_split: numpy array to split (self.sites_index or self.samples_index)
    """
    random = np.random.multinomial(
        n=len(to_split) - n_block,
        pvals=np.ones(n_block) / n_block,
    ) + 1

    # index of sites in each block
    index = np.split(to_split, np.cumsum(random))[0: n_block]
    # number of sites in each fold
    sizes = list(map(len, index))
    return index, np.array(sizes)


class Observation1:
    """observation of one population: for estimating the site diversity
    """
    def __init__(self, pop_ts, num_inds, max_sites):
        self.ts = sim.observe(pop_ts, num_inds, max_sites, num_pop=1)
        self.num_samples = self.ts.num_samples
        self.num_sites = self.ts.num_sites
        self.samples_index = np.arange(self.num_samples)
        self.sites_index = np.arange(self.num_sites)
        self.site_diversity = self.ts.diversity(span_normalise=False, windows='sites')
        self.genos = self.ts.genotype_matrix()

    def get_site_diversity(self, samples_index=None):
        """returns average pairwise diversity of a set of samples across a set of sites.
        Used for resampling over samples. This function is general in the sense that samples may have duplicates.
        Sites_index is the same as the self
        @samples_index = the resampled samples index (may contain duplicate)
        """
        if samples_index is None:
            samples_index = self.samples_index
        num_samples = len(samples_index)
        ac = self.genos[self.sites_index, :][:, samples_index].sum(1)
        num_pairs = int(num_samples * (num_samples - 1) / 2)
        n_different_pairs = ac * (num_samples - ac)
        return np.mean(n_different_pairs / num_pairs)

    def bootstrap_sites_diversity(self, num_boot=500):
        """bootstrap resampling over sites
        @num_boot: num of bootstrap times (default 500)
        """
        weights = np.random.multinomial(
            n=self.num_sites,
            pvals=np.ones(self.num_sites) / self.num_sites,
            size=num_boot
        )
        return np.mean(self.site_diversity * weights, axis=1)

    def bootstrap_samples_diversity(self, num_boot=500):
        """bootstrap over samples
        @num_boot: num of bootstrap times (default 500)
        """
        pool = Pool(processes=cpu_count())
        inputs = [np.random.choice(self.num_samples, self.num_samples, replace=True) for _ in range(num_boot)]
        resample_values = pool.map(self.get_site_diversity, inputs)
        pool.close()
        pool.join()
        return resample_values

    def jackknife_one_sites_diversity(self):
        """delete one jackknife resample over sites
        """
        weights = np.ones((self.num_sites, self.num_sites), dtype=int)
        np.fill_diagonal(weights, 0)
        return (self.site_diversity * weights).sum(axis=1) / (self.num_sites - 1)

    def jackknife_one_samples_diversity(self):
        """delete one jackknife resample over samples
        """
        pool = Pool(processes=cpu_count())
        inputs = [np.delete(self.samples_index, i) for i in self.samples_index]
        resample_values = pool.map(self.get_site_diversity, inputs)
        pool.close()
        pool.join()
        return resample_values

    def jackknife_mj_sites_diversity(self):
        """delete_mj jackknife resampling methods over sits with unequal sizes
        """
        n_block = int(np.sqrt(self.num_sites))
        index, sizes = jk_split(n_block, self.sites_index)

        weights = np.ones((n_block, self.num_sites), dtype=int)

        # fill deleted block with 0
        for i, indices in enumerate(index):
            weights[i][indices] = 0

        return (self.site_diversity * weights).sum(axis=1) / ((weights == 1).sum(axis=1)), sizes

    def jackknife_mj_samples_diversity(self):
        """delete_mj jackknife resampling methods over samples with unequal sizes
        """
        n_block = int(np.sqrt(self.num_samples))
        index, sizes = jk_split(n_block, self.samples_index)

        pool = Pool(processes=cpu_count())
        inputs = [np.delete(self.samples_index, i) for i in index]
        resample_values = pool.map(self.get_site_diversity, inputs), sizes
        pool.close()
        pool.join()
        return resample_values


class Observation2:
    """observation of two population: for estimating fst
    """
    def __init__(self, pop_ts, num_inds, max_sites):
        self.ts = sim.observe(pop_ts, num_inds, max_sites, num_pop=2)
        self.popA = self.ts.sample(population=0)
        self.popB = self.ts.sample(population=1)
        self.sites_index = np.arange(self.ts.num_sites)

        self.__ga = allel.GenotypeArray(self.ts.genotype_matrix().reshape(self.ts.num_sites, self.ts.num_samples, 1),
                                        dtype='i1')
        self.__num, self.__denom = allel.hudson_fst(self.__ga[:, self.popA, :].count_alleles(),
                                                    self.__ga[:, self.popB, :].count_alleles())
        self.__popA_num_inds = int(len(self.popA) / 2)
        self.__popB_num_inds = int(len(self.popB) / 2)

        self.fst = np.sum(self.__num) / np.sum(self.__denom)

    def get_fst_general(self, params):
        """returns Hudson's Fst: used for resampling over samples

        This function is general in the sense that all of:
          (popA_samples, popB_samples) may have duplicates.
        @params: wrapped for multiprocess
            @popA_samples = the samples from the first population to be used
            @popB_samples = the samples from the second population to be used
        """
        # count alleles within each population at the selected sites and inds
        popA_samples, popB_samples = params
        ac1 = self.__ga[self.sites_index][:, popA_samples, :].count_alleles()
        ac2 = self.__ga[self.sites_index][:, popB_samples, :].count_alleles()
        # calculate Hudson's Fst (weighted)
        num, denom = allel.hudson_fst(ac1, ac2)
        return np.sum(num) / np.sum(denom)

    def bootstrap_samples_fst(self, num_boot=500):
        """Calculate Fst while bootstrap resampling over individuals.
        uses multiprocessing.Pool to run across multiple cores.
        @num_boot = number of bootstrap times
        """
        inputs = [(sim.sample_individuals(self.popA, self.__popA_num_inds, replace=True),
                   sim.sample_individuals(self.popB, self.__popB_num_inds, replace=True),
                   ) for _ in range(num_boot)]
        return self.__pool.map(self.get_fst_general, inputs)

    def jackknife_samples_fst(self):
        """
        Calculate Fst while jackknife resampling over individuals.
        uses multiprocessing.Pool to run across multiple cores
        """
        inputs = [(sim.sample_individuals(self.popA, self.__popA_num_inds, replace=True), self.popB)
                  for _ in range(self.__popA_num_inds)] + \
                 [(self.popA, sim.sample_individuals(self.popB, self.__popB_num_inds, replace=True))
                  for _ in range(self.__popB_num_inds)]

        return self.__pool.map(self.get_fst_general, inputs)

    def bootstrap_sites_fst(self, num_boot):
        """Calculate Fst for bootstrap of sites.
        @num_boot = number of bootstrap re-samplings
        """
        weights = np.random.multinomial(
            n=self.ts.num_sites,
            pvals=np.ones(self.ts.num_sites) / self.ts.num_sites,
            size=num_boot
        )
        return (self.__num * weights).sum(1) / (self.__denom * weights).sum(1)

    def jackknife_sites_fst(self):
        """Calculate Fst for jackknife of sites.
        """
        num_sum, denom_sum = self.__num.sum(), self.__denom.sum()
        return (num_sum - self.__num) / (denom_sum - self.__denom)

    # TODO: jackknife_mj_sites_fst
