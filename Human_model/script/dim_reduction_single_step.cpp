#include<bits/stdc++.h>
using namespace std;

#define i64 long long int
mt19937 rnd(741);
int _koala_gap = 0;

// 读样本信息
struct Info {
	string seq_id;
	int label;
	char stage[9];
	float stage_num;
	int sex, age;
};
map<string, Info> info_dict;
void load_sample_info(string input_info) {
	puts("Loading sample info...");
	FILE *f_info = fopen(input_info.c_str(), "r");
	
	int n;
	fscanf(f_info, "%d", &n);
	for(int i=0;i<n;i++) {
		Info info;
		char seq_id_arr[33];
		fscanf(f_info, "%s%d%s%f%d%d", 
			seq_id_arr, 
			&info.label, 
			info.stage, 
			&info.stage_num, 
			&info.sex, 
			&info.age
		);
		info.seq_id = seq_id_arr;
		info_dict[info.seq_id] = info;
	}
	printf("Total Size:%d\n",info_dict.size());	
	fclose(f_info);
}

char __line__[2002002];
vector<int> label;
vector<string> get_seq_id(string file_name){
	vector<string> ret;
	FILE* inp = fopen(file_name.c_str(), "r");

	if(NULL == inp) {
		printf("FILE_ERROR %s\n",file_name.c_str());
		return ret;
	}
    fgets(__line__,2002000,inp);
	bool ff = false;
	string cur = "";
	for(char*c = __line__; !(*c == 0 || *c == '\n'); c++){
		if(*c == '\''){
			if(ff){
				int L = cur.length();
				if(L >= 3 && cur.substr(L-3, 3) == "bam"){
					string o = "";
					for(char C : cur){
						if(C == '.')break;
						o += C;
					}
					ret.push_back(o);
				}
			}
			ff = !ff;
			cur = "";
		}else
		if(ff)
			cur += *c;
	}
	fclose(inp);
	return ret;
}
void load_sample_label(string input_tab) {
	vector<string> ids = get_seq_id(input_tab);
	label.clear();
	for(auto it = ids.begin(); it!=ids.end();it++) {
		string seq_id = (*it);
		if(info_dict.find(seq_id) != info_dict.end())
			label.push_back(info_dict[seq_id].label);
		else {
			label.push_back(-1);
		}
	}
}


pair<int, double> get_score(vector<pair<double, int> > v) {
	pair<int, double> res = make_pair(0, 0);
	sort(v.begin(), v.end());

	if (v[0].first + 0.1 >= v.back().first) return res; 
	int len = (int)v.size();
	int tot_pos = 0, tot_neg = 0;
	for(int i=0;i<len;i++) {
		if (v[i].second == 0) tot_neg ++;
		else if (v[i].second == 1) tot_pos ++;
	}
	int cur_pos = 0, cur_neg = 0;
	for(int i=0;i<len-1;i++) {
		if (v[i].second == 0) cur_neg ++;
		else if (v[i].second == 1) cur_pos ++;
		
		int score = max(
			cur_neg + (tot_pos-cur_pos),
			cur_pos + (tot_neg-cur_neg)
		);
		double gap = v[i+1].first - v[i].first;
		if (gap<_koala_gap) continue;
		if (gap<1e-6) continue;
		if (score > res.first || score == res.first && gap > res.second)
			res = make_pair(score, gap);
	}
	return res;
}
void upd(int &ans_sum, int &ans_gap, int sum, int gap) {
	if (sum < ans_sum) return;
	if (sum == ans_sum && gap > ans_gap) return;
	ans_sum = sum;
	ans_gap = gap;
}

vector<double> __tmp__;
void __fast_read__(FILE* inp, vector<double> &v) {
	__line__[0] = 0;
	fgets(__line__,2002000,inp);
	char*c = __line__;
	bool ff = false, dec = false;
	double cur = 0., w = 1.;
	v.clear();
	while(1){
		if(*c=='.') {
			dec = true;
			w/=10;
		} else if(isalnum(*c)){
			if(isdigit(*c)){
				if(!ff)
					cur = *c - '0';
				else if(dec) {
					cur += w*(*c - '0');
					w /= 10;
				}
				else
					cur = cur * 10 + *c-'0';
			}else{
				switch(*c){
				case 'X': cur = 23; break;
				case 'Y': cur = 24; break;
				case 'M': cur = 25; break;
				default: cur = -1; break;
				}
			}
			ff = true;
		}else{
			if(ff) v.push_back(cur);
			ff = dec = false;
			w = 1;
		}
		if(*c == 0 || *c == '\n')return;
		c ++;
	}
}
void load_detail(string input_tab, string output_bed){
	vector<pair<double, int> > v(0);
	int sz = (int)label.size();
	for(int i=0;i<sz;i++)
		v.push_back(make_pair(0, label[i]));
	
	FILE* inp = fopen(input_tab.c_str(), "r");
	if(NULL == inp)return;
	
	FILE* ou = fopen(output_bed.c_str(), "w");
	
	fgets(__line__,2002000,inp);
	while(1) {
		__fast_read__(inp, __tmp__);
		if(!__tmp__.size()) break;

		for(int i=0;i<sz;i++) v[i].first = __tmp__[i+3];
		auto res = get_score(v);
		
		double eps = 1e-6;
		int chr_id = int(__tmp__[0]+eps);
		if(chr_id<=22)fprintf(ou, "%d", chr_id);
		else fprintf(ou, "%c", chr_id==23?'X':chr_id==24?'Y':'M');
		fprintf(ou, "\t%d\t%d\t%d\t%.4lf\n", int(__tmp__[1]+eps), int(__tmp__[2]+eps), res.first, res.second);
	}
	
	fclose(inp);
	fclose(ou);
}

int main(int argc,char*argv[]) {
	puts("usage: ./dim_reduction_single_step tab_id");
	
	string input_info = "./all.sample.info";
	string input_tab = "./" + string(argv[1]);
	string output_bed = "./" + string(argv[1]) + ".bed";

	puts("======= dim selection ======");
	puts("Expect runtime: 20s");
	load_sample_info(input_info);
	load_sample_label(input_tab);
	load_detail(input_tab, output_bed);
	puts("Load complete.");
	printf("%d\n", int(clock() / CLOCKS_PER_SEC));

	return 0;
}