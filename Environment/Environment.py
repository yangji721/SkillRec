
class Environment(object):
    def __init__(self, lambda_d, d_estimator, job_matcher, n_skill):
        self.lambda_d = lambda_d
        self.d_estimator = d_estimator
        self.job_matcher = job_matcher
        self.n_skill = n_skill
        self.state = [0] * self.n_skill
        self.state_list = []
        self.target_list = []

    def set_target(self, target):
        self.target_list = target

    def get_reward(self, easy, salary):
        return self.lambda_d * easy + salary

    def clear(self):
        self.job_matcher.reset()
        self.d_estimator.clear()
        self.state = [0] * self.n_skill
        self.state_list = []

    def add_prefix(self, prefix):
        for s in prefix:
            self.add_skill(s, evaluate=False)
        salary_start = self.job_matcher.top_average()
        return salary_start[0]

    def add_skill(self, s, evaluate=True):
        self.job_matcher.add(int(s))
        easy = self.d_estimator.predict_and_add(int(s))
        salary, salary_base, similarity_base = -1, -1, -1
        if evaluate:
            salary_sum = self.job_matcher.top_average()
            salary = salary_sum[0]
            salary_base = salary_sum[1]
            similarity_base = salary_sum[2]
        self.state[s] = 1
        self.state_list.append(s)
        return easy, salary, self.get_reward(easy, salary), salary_base, similarity_base
