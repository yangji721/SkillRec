import sys
sys.path.append("../../")
sys.path.append("../")
from Sampler import DistributionSampler
from Utils.JobReader import n_skill, sample_info, read_offline_samples, itemset_process
from Utils.Utils import read_pkl_data
from Utils.Functions import evaluate
from Environment.DifficultyEstimatorGLinux import DifficultyEstimator
from Environment.Environment import Environment
from Environment.JobMatcherLinux import JobMatcher
from CONFIG import HOME_PATH

if __name__ == "__main__":
    direct_name = "resume"

    env_params = {"lambda_d": 0.1, "beta": 0.1, 'pool_size': 100}
    mode = "frequency"
    #mode = "uniform"
    T = 20
    sample_lst, skill_cnt, _ = sample_info()

    itemset = itemset_process(skill_cnt)

    d_estimator = DifficultyEstimator(item_sets=[u[0] for u in itemset], item_freq=[u[1] for u in itemset], n_samples=len(sample_lst))
    job_matcher = JobMatcher(n_top=100, skill_list=[u[0] for u in sample_lst], salary=[u[1] for u in sample_lst], w=5, th=10.0 / 9)
    environment = Environment(lambda_d=env_params['lambda_d'], d_estimator=d_estimator, job_matcher=job_matcher, n_skill=n_skill)

    if mode == 'uniform':
        # 均匀分布
        uniform_p = [1.0 / n_skill] * n_skill
        sampler = DistributionSampler(p=uniform_p, n_a=n_skill)
    elif mode == 'frequency':
        cnt_sum = sum(skill_cnt)
        freq_p = [u * 1.0 / cnt_sum for u in skill_cnt]
        sampler = DistributionSampler(p=freq_p, n_a=n_skill)
    #elif mode == 'salary':
    #    sampler = BestSalarySampler(job_matcher, n_skill, skill_cnt)
    #else:
    #    sampler = BestShortSampler(environment, n_skill, skill_cnt) # 太慢了，换一种方式

    train_samples = read_offline_samples(direct_name) # skill_lst, r_lst
    data_test = read_pkl_data(HOME_PATH + "data/%s/train_test/testdata.pkl" % direct_name)

    evaluate(sampler, environment, data_test, train_samples, -1, T=T)
