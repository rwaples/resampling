"""This file contains methods for generating population data
as well as observation data
"""
import numpy as np
import msprime


def sim_one_population(diploid_size, seq_len, rec_rate, mut_rate, seed=None):
    """simulate two populations that have diverged from a common ancestral population.
    Returns a tree sequence.
    @diploid_size = the population size of each population,
        also the size of the ancestral population
    @seq_len = length of the genome, units ~ base-pairs
    @rec_rate = recombination rate, units = rate per bp, per generation
    @mut_rate = mutation rate, units = rate per bp, per generation
    """
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=diploid_size)

    ts = msprime.sim_ancestry(
        samples={'A': diploid_size},  # diploid samples
        demography=demography,
        ploidy=2,
        sequence_length=seq_len,
        discrete_genome=False,
        recombination_rate=rec_rate,
        model='dtwf',
        random_seed=seed
    )

    ts = msprime.sim_mutations(
        ts,
        rate=mut_rate,
        discrete_genome=False,
        random_seed=seed
        )

    return ts


def sim_population(diploid_size, split_time, seq_len, rec_rate, mut_rate, seed=None):
    """simulate two populations that have diverged from a common ancestral population.
    Returns a tree sequence.
    @diploid_size = the population size of each population,
        also the size of the ancestral population
    @split_time = current populations split from the
        ancestral population this many generations ago
    @seq_len = length of the genome, units ~ base-pairs
    @rec_rate = recombination rate, units = rate per bp, per generation
    @mut_rate = mutation rate, units = rate per bp, per generation
    """
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=diploid_size)
    demography.add_population(name="B", initial_size=diploid_size)
    demography.add_population(name="C", initial_size=diploid_size)
    demography.add_population_split(time=split_time,
                                    derived=["A", "B"], ancestral="C")

    ts = msprime.sim_ancestry(
        samples={'A': diploid_size, 'B': diploid_size},  # diploid samples
        demography=demography,
        ploidy=2,
        sequence_length=seq_len,
        discrete_genome=False,
        recombination_rate=rec_rate,
        model='dtwf',
        random_seed=seed
    )

    ts = msprime.sim_mutations(
        ts,
        rate=mut_rate,
        discrete_genome=False,
        start_time=split_time,
        random_seed=seed
        )

    return ts


def sample_individuals(haploid_indexes, n, replace, seed=None):
    """
    return the (haploid) indexes that correspond to
    taking n diploid samples from the supplied haploid indexes

    @haploid_indexes = contiguous indexes for haploids
        should be from a single population.
    @n = the number of diploid individuals to take.
    """

    # ensure the haploid indexes are consecutive
    # diff = np.diff(haploid_indexes)
    # assert np.sum(diff == 1) == (len(haploid_indexes)-1)

    ind_indexes = haploid_indexes[::2]
    np.random.seed(seed)
    ind_samples = np.sort(np.random.choice(ind_indexes, n, replace=replace))
    haploid_samples = np.zeros(len(ind_samples)*2, dtype='int')
    haploid_samples[0::2] = ind_samples
    haploid_samples[1::2] = ind_samples+1
    return haploid_samples
    
    
def observe(ts, num_ind, max_sites, num_pop, seed=None):
    """observe num_ind diploids from each population
    simplify the ts, removing non-variable sites across those individuals

    @ts = tree-sequence
    @num_ind = number of diploids to sample from each population
    @max_sites = retain at most max_sites, from among variable sites
    @num_pop = number of population
    """
    pop = [ts.samples(population=i) for i in range(num_pop)]
    pop_ind = [sample_individuals(i, num_ind, replace=False, seed=seed) for i in pop]
    all_ind = np.concatenate(pop_ind)
    obs_ts = ts.simplify(samples=all_ind, filter_sites=False)

    if obs_ts.num_sites > max_sites:
        all_sites = np.arange(len(obs_ts.sites()))
        np.random.seed(seed)
        sites_keep = np.random.choice(all_sites, max_sites, replace=False)
        sites_remove = np.setdiff1d(all_sites, sites_keep)
        obs_ts = obs_ts.delete_sites(sites_remove)

    return obs_ts
