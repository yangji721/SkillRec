#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <cmath>
#include <cstring>
#include <iterator>
#include <algorithm>
using namespace std;

class State {
public:
	int samp_id;
	double salary;
	double similarity_base;
	double salary_base;
	State(int sampid = 0, double salary = 0, double salary_base = 0, double similarity_base = 0);
	friend bool operator < (State a, State b);
	friend bool operator == (State a, State b);
};
class JobMatcher {
    private:
		int w, n_samples, n_top;
        double th, th2, w_a, w_b;
		vector<int> preference;
		vector<vector<int>> skill_list;
		vector<double> salary;
		vector<int> skill_sample_list[2000];
		int p[600000] = { 0 };
		priority_queue<State> Qs;
    public:
		JobMatcher(int w, double th, double th2, double w_a, double w_b, int n_top, vector<vector<int> > skill_list, vector<double> salary);
		void reset();
		void set_target(vector<int> target);
		double calc_similarity(int s_num);
		double discount(int n_cur, int n_all);
		void add(int s);
		double predict_salary(int s);
		double top_average_of_heap(priority_queue<State> Qs_tmp);
		vector<double> top_average();
};
