import site_diversity
import fst
from datetime import datetime


if __name__ == '__main__':
    prefix = datetime.now().strftime("%m%d%s")
    site_diversity_df = site_diversity.experiment(num_exp=2, num_obs=100, confidence=0.95)
    site_diversity_df.to_csv(f'../data/{prefix}_site_diversity.csv', index=False)

    fst_df = fst.experiment(num_exp=2, num_obs=100, confidence=0.95)
    fst_df.to_csv(f'../data/{prefix}_fst.csv', index=False)
