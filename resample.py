import numpy as np
from sklearn.model_selection import KFold


def site_diversity(ts):
    """return a numpy array of each site's diversity across the observed population
    
    """
    samples = np.arange(ts.num_samples)
    sites_index = np.arange(ts.num_sites)
    nsamples = len(samples)
    genos = ts.genotype_matrix()
    
    ac = genos[sites_index, :][:,samples].sum(1)
    
    npairs = int(nsamples * (nsamples - 1) / 2)  # total numbers of pairs of samples
    n_different_pairs = ac * (nsamples - ac) # number of those pairs that have different alleles
    return (n_different_pairs / npairs)
    

def bt_resample_sites(params):
    """
    Bootstrap resampling over sites 
    return a numpy array containing site_diversity of each bootstrap
    """
    
    diversity, n_boot = params
    
    num_sites = len(diversity)
    bt_weights = np.random.multinomial(
        n=num_sites,
        pvals=np.ones(num_sites)/num_sites,
        size=n_boot
    )
    bt_vals = np.mean((diversity * bt_weights), axis=1)
    
    return bt_vals
    
    
def jk_resample_sites(diversity, obs_ts_diversity):
    '''
    Jackknife resamle over sites
    Return a numpy array containing the psedovalues of site diversity
    '''
    
    num_sites = len(diversity)
    jk_weights = np.ones((num_sites, len(diversity)), dtype=int)
    np.fill_diagonal(jk_weights, 0)

    jk_vals = num_sites * obs_ts_diversity - \
                (num_sites - 1) * np.mean((diversity * jk_weights), axis=1)
    
    return jk_vals


def jk_block_resample_sites(diversity, obs_ts_diversity, n_fold):
    '''
    Jackknife Block resamle over sites
    Return a numpy array containing the psedovalues of site diversity
    '''
    
    num_sites = len(diversity)
    jk_weights = np.ones((n_fold, num_sites), dtype=int)
    kf = KFold(n_splits=n_fold, shuffle=True)
    
    # fill deleleted block with 0
    for i, (_, zero_index) in enumerate(kf.split(np.arange(num_sites))):
        jk_weights[i][zero_index] = 0

    jk_vals = n_fold * obs_ts_diversity - \
                (n_fold - 1) * np.mean((diversity * jk_weights), axis=1)
    
    return jk_vals
    
    
def resample_sites(diversity, obs_ts_diversity, n_boot, n_fold):
    '''
    resample sites for bootstrap, jackknife and jackkife block 
    @ts: observed population
    '''
    
    res = {}
    
    #res['bt_sites'] = bt_resample_sites(diversity, n_boot)
    #res['jk_sites'] = jk_resample_sites(diversity, ts.num_sites, obs_ts_diversity)
    #res['jk_block_sites'] = jk_block_resample_sites(diversity, obs_ts_diversity, ts.num_sites, n_fold)
    
    return bt_resample_sites(diversity, n_boot)