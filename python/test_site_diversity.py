import unittest
import numpy as np
import simulation as sim
from observation import Observation1


class TestResampleMethods(unittest.TestCase):
    def setUp(self):
        self.pop_ts = sim.sim_one_population(
            diploid_size=200,
            seq_len=1e9,
            rec_rate=1e-8,
            mut_rate=1e-8
        )
        self.num_inds, self.max_sites = 100, 1000
        self.obs_ts = Observation1(self.pop_ts, self.num_inds, self.max_sites)

    # test obs_ts values
    def test_obs(self):
        self.assertEqual(self.obs_ts.num_sites, self.max_sites)
        self.assertEqual(self.obs_ts.num_samples, self.num_inds * 2)
        self.assertEqual(self.obs_ts.genos.shape, (self.max_sites, self.num_inds * 2))

    # test get_site_diversity function
    def test_get_site_diversity(self):
        self.assertEqual(self.obs_ts.site_diversity.mean(), self.obs_ts.get_site_diversity())
        self.assertEqual(self.obs_ts.site_diversity.mean(), self.obs_ts.get_site_diversity(self.obs_ts.samples_index))

    # test bootstrap functions with default number of bootstrap
    def test_bootstrap(self):
        bt_sites = self.obs_ts.bootstrap_sites_diversity()
        bt_samples = self.obs_ts.bootstrap_samples_diversity()
        self.assertEqual(len(bt_sites), 500)
        self.assertEqual(len(bt_samples), 500)

        # test bootstrap functions with input of num_boot
        num_boot = 100
        bt_sites_100 = self.obs_ts.bootstrap_sites_diversity(num_boot)
        bt_samples_100 = self.obs_ts.bootstrap_samples_diversity(num_boot)
        self.assertEqual(len(bt_sites_100), 100)
        self.assertEqual(len(bt_samples_100), 100)

    # test jackknife delete one functions
    def test_jackknife_one(self):
        jk_one_sites = self.obs_ts.jackknife_one_sites_diversity()
        jk_one_samples = self.obs_ts.jackknife_one_samples_diversity()
        self.assertEqual(len(jk_one_sites), self.obs_ts.num_sites)
        self.assertEqual(len(jk_one_samples), self.obs_ts.num_samples)

    # test jackknife delete mj functions
    def test_jackknife_mj(self):
        n_block_sites = int(np.sqrt(self.obs_ts.num_sites))
        n_block_samples = int(np.sqrt(self.obs_ts.num_samples))
        jk_mj_sites, sites_sizes = self.obs_ts.jackknife_mj_sites_diversity()
        jk_mj_samples, samples_sizes = self.obs_ts.jackknife_mj_samples_diversity()
        self.assertEqual(len(jk_mj_sites), n_block_sites)
        self.assertEqual(len(sites_sizes), n_block_sites)
        self.assertEqual(len(jk_mj_samples), n_block_samples)
        self.assertEqual(len(samples_sizes), n_block_samples)


if __name__ == '__main__':
    unittest.main(verbosity=2)
