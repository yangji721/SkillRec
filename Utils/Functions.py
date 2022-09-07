from Utils.JobReader import skill_lst, n_skill
from Utils.Utils import sparse_to_dense
import time


def evaluate(sampler, environment, data_test, train_samples, epoch, T=16, verbose=False):
    time_start = time.time()

    N_test = len(data_test)
    step_salary, step_easy, step_r, step_salary_base, step_similarity_base = [0] * min(20, T * 10), [0] * min(20, T * 10), [0] * min(20, T * 10), [0] * min(20, T * 10), [0] * min(20, T * 10)
    avg_easy, avg_salary, avg_salary_base, avg_similarity_base = 0, 0, 0, 0 
    avg_start = 0
    for x, y in data_test:
        # 初始化
        environment.clear()
        step_metrics = []
        ### General setting for evaluation
        # prefix = train_samples[x][0][0]
        # preference = train_samples[x][0][1]
        
        ### Case study 2
        prefix = [160, 671, 89, 856, 843, 631, 779, 552, 301, 434, 581, 212]
        # preference = [6 ,169, 812, 846, 953, 958]
        preference =  [57, 301, 631, 701, 746, 843, 846, 956]

        ### Case study 1
        # prefix = [248, 383, 502, 507, 581, 812, 828, 846, 882, 953, 985]
        # preference = [6, 169, 434, 812, 843, 846, 882, 958]

        environment.job_matcher.set_target(preference)

        salary_start = environment.add_prefix(prefix)
        prefix_names = [skill_lst[u] for u in prefix]
        append_skills = []
        for t in range(T):
            s, q = sampler.sample_with_preference(sparse_to_dense(preference, n_skill), environment.state)
            easy, salary, r, salary_base, similarity_base = environment.add_skill(s, evaluate=True)
            step_metrics.append((r, easy, salary, salary_base, similarity_base, q))
            append_skills.append(skill_lst[s])
        if verbose:
            print(prefix_names, append_skills)

        print(prefix_names, append_skills)
        N_test = 1
        # 评价指标
        avg_start += salary_start / N_test
        avg_easy += sum([val[1] for val in step_metrics]) / T / N_test
        avg_salary += sum([val[2] for val in step_metrics]) / T / N_test
        avg_salary_base += sum([val[3] for val in step_metrics]) / T / N_test
        avg_similarity_base += sum([val[4] for val in step_metrics]) / T / N_test

        for j, k in enumerate(range(1, T + 1)):
            step_r[j] += step_metrics[k - 1][0] / N_test
            step_easy[j] += step_metrics[k - 1][1] / N_test
            step_salary[j] += step_metrics[k - 1][2] / N_test
            step_salary_base[j] += step_metrics[k - 1][3] / N_test
            step_similarity_base[j] += step_metrics[k - 1][4] / N_test
        
        break

    time_end = time.time()
    sec = time_end - time_start
    # 输出
    print("[time:%d, epoch %d] avg_easy: %.4f, avg_salary: %.4f, salary_start: %.4f , avg_salary_base: %.4f, avg_similarity_base: %.4f , step_salary: " % (sec, epoch, avg_easy, avg_salary, avg_start, avg_salary_base, avg_similarity_base), end="")

    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_salary[j], end="")
    print("\n step_easy: ", end="")

    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_easy[j], end="")
    
    print("\n step_salary_base: ", end="")
    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_salary_base[j], end="")

    print("\n step_similarity_base: ", end="")
    for j, k in enumerate(range(1, T + 1)):
        print("%.4f," % step_similarity_base[j], end="")
    print("")


def evaluate_step(sampler, environment, data_test, train_samples, T=16):
    step_easy_lst, step_salary_lst = [], []
    salary_lst = []
    for x, y in data_test:
        step_salary, step_easy = [], []
        # 初始化
        environment.clear()
        prefix = train_samples[x][0][:y]
        salary_lst.append(environment.add_prefix(prefix))
        append_skills = []
        for t in range(T):
            s, q = sampler.sample(environment.state)
            easy, salary, r = environment.add_skill(s, evaluate=True)
            step_easy.append(easy)
            step_salary.append(salary)
            append_skills.append(skill_lst[s])
        step_easy_lst.append(step_easy)
        step_salary_lst.append(step_salary)
    return salary_lst, step_easy_lst, step_salary_lst
