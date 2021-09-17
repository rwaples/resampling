import div
import fst
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    """
    prefix = datetime.now().strftime("%m%d%s")
    site_diversity_df = div.experiment(num_exp=2, num_obs=100, confidence=0.95)
    site_diversity_df.to_csv(f'../data/{prefix}_site_diversity.csv', index=False)

    fst_df = fst.experiment(num_exp=2, num_obs=100, confidence=0.95)
    fst_df.to_csv(f'../data/{prefix}_fst.csv', index=False)
    """
    np.random.seed(1)
    for i in range(2):
        print(np.random.randint(1, 5, 2))
