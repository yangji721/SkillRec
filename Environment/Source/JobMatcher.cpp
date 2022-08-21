#include "JobMatcher.h"

JobMatcher::JobMatcher(int w, double th, double th2, double w_a, double w_b, int n_top, vector<vector<int> > skill_list, vector<double> salary) {
	this->w = w;
	this->th = th;
	this->th2 = th2;
	this->w_a = w_a;
	this->w_b = w_b;
	this->n_samples = skill_list.size();
	this->n_top = n_top;
	for (int i = 0; i < this->n_samples; i++) {
		this->skill_list.push_back(vector<int>());
		for (int j = 0; j < skill_list[i].size(); j++) {
			this->skill_list[i].push_back(skill_list[i][j]);
		}
	}
	for (int i = 0; i < n_samples; i++) {
		this->salary.push_back(salary[i]);
		for (int j = 0; j < this->skill_list[i].size(); j++) {
			int s = this->skill_list[i][j];
			this->skill_sample_list[s].push_back(i);
		}
	}
}

void JobMatcher::reset() {
	memset(p, 0, sizeof(p));
	while (!Qs.empty()) Qs.pop();
	similarity.clear();
}

// set similarity_score
void JobMatcher::set_target(vector<int> target) {
	sort(target.begin(), target.end());
	for (int i = 0; i < n_samples; i++) {
		auto temp = skill_list[i];
		sort(temp.begin(), temp.end());
		vector<int> intersection_set, target_diff, job_diff;
        set_intersection(target.begin(), target.end(), temp.begin(), temp.end(), back_inserter(intersection_set));
		set_difference(target.begin(), target.end(), temp.begin(), temp.end(), inserter(target_diff, target_diff.begin()));
		set_difference(temp.begin(), temp.end(), target.begin(), target.end(), inserter(job_diff, job_diff.begin()));
		double similarity_score = double(intersection_set.size()) / double(intersection_set.size() + w_a*target_diff.size() + w_b * job_diff.size());
		this->similarity.push_back(similarity_score);
	}
}

double JobMatcher::discount(int n_cur, int n_all) {
	double r = n_cur * 1.0 / n_all;
	return 1.0 / (1.0 + exp(-w * (r - 0.5)));
}

void JobMatcher::add(int s) {
	for (int i = 0; i < skill_sample_list[s].size(); i++) {
		int u = skill_sample_list[s][i];
		p[u] ++;
		if (p[u] * th < skill_list[u].size()) continue;
		// add similarity threshold
		if (similarity[u] < th2) continue;
		double r = discount(p[u], skill_list[u].size());
		//modify the salary mechanism
		Qs.push(State(u, (r + similarity[u]) * salary[u], r * salary[u], similarity[u] * salary[u]));
	}
}

double JobMatcher::predict_salary(int s) {
	priority_queue <State> Qs_tmp;
	vector<State> tmp;
	while (!Qs.empty() && tmp.size() < n_top) {
		State u = Qs.top();
		Qs.pop();
		tmp.push_back(u);
		Qs_tmp.push(u);
	}
	while (!Qs.empty()) Qs.pop();
	for (int i = 0; i < tmp.size(); i++) Qs.push(tmp[i]);

	for (int i = 0; i < skill_sample_list[s].size(); i++) {
		int u = skill_sample_list[s][i];
		int pnow = p[u] + 1;
		if (pnow * th < skill_list[u].size()) continue;
		// add similarity threshold
		if (similarity[u] < th2) continue;
		double r = discount(pnow, skill_list[u].size());
		//modify the salary mechanism
		Qs_tmp.push(State(u, (r + similarity[u]) * salary[u], r * salary[u], similarity[u] * salary[u]));
	}
	return top_average_of_heap(Qs_tmp);
}

double JobMatcher::top_average_of_heap(priority_queue<State> Qs_tmp){
	double ret = 0;
	int num = n_top;
	set<int> S;
	while (num && !Qs_tmp.empty()) {
		State s = Qs_tmp.top();
		Qs_tmp.pop();
		if (S.find(s.samp_id) == S.end()) {
			S.insert(s.samp_id);
			ret += s.salary / n_top;
			num--;
		}
	}
	return ret;
}

vector<double> JobMatcher::top_average() {
	double ret = 0;
	double salary_ret = 0;
	double similarity_ret = 0;
	int num = n_top;
	int ck = 0;
	State a[1000];
	set<int> S;
	while (num && !Qs.empty()) {
		State s = Qs.top();
		Qs.pop();
		if (S.find(s.samp_id) == S.end()) {
			S.insert(s.samp_id);
			a[ck++] = s;
			ret += s.salary / n_top;
			salary_ret += s.salary_base / n_top;
			similarity_ret += s.similarity_base / n_top;
			num--;
		}
	}
	while (!Qs.empty()) Qs.pop();
	for (int i = 0; i < ck; i++) Qs.push(a[i]);
	vector<double> result = {ret, salary_ret, similarity_ret};
	return result;
}

State::State(int sampid, double salary, double salary_base, double similarity_base){
	this->samp_id = sampid;
	this->salary = salary;
	this->salary_base = salary_base;
	this->similarity_base = similarity_base;
}

bool operator< (State a, State b) {
	if (a.salary != b.salary) {
		return a.salary < b.salary;
	}
	else {
		return a.samp_id < b.samp_id;
	}
}
bool operator == (State a, State b) {
	return a.salary == b.salary && a.samp_id == b.samp_id;
}
