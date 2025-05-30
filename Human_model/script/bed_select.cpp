#include<bits/stdc++.h>
using namespace std;

struct Node {
	int chr, st, ed;
	double sc1, sc2, sc3;
	pair<int, pair<int, int > > id;
	Node() {}
	Node(int _chr, int _st, int _ed, double _sc1, double _sc2, double _sc3) {
		chr = _chr;
		st = _st;
		ed = _ed;
		sc1 = _sc1;
		sc2 = _sc2;
		sc3 = _sc3;
		id = make_pair(chr, make_pair(st, ed));
	}
	string chr_str() {
		if(chr==23) return "X";
		if(chr==24) return "Y";
		if(chr==25) return "M";
		return to_string(chr);
	}
};
vector<Node> v;

char __line__[2002002];
vector<double> __tmp__;
double eps = 1e-9;
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

int cmp(const Node &a, const Node &b) {
	if(a.sc1 != b.sc1) return a.sc1 > b.sc1;
	return a.sc2 > b.sc2;
}

bool intersect(const pair<int, pair<int, int > > &a, const pair<int, pair<int, int > > &b) {
	if(a.first != b.first) return false;
	int al = a.second.first, ar = a.second.second;
	int bl = b.second.first, br = b.second.second;
	if(ar <= bl) return false;
	if(br <= al) return false;
	return true;
}

void init(FILE* inp) {
	v.clear();
	char s[3]; int st,ed; double sc1,sc2;
	while(1) {
		__fast_read__(inp, __tmp__);
		if(!__tmp__.size()) break;
		v.push_back(Node(int(__tmp__[0]+eps), int(__tmp__[1]+eps), int(__tmp__[2]+eps), __tmp__[3], __tmp__[4], __tmp__[5]));
	}
	printf("read finish: %d\n", int(clock() / CLOCKS_PER_SEC));
	
	sort(v.begin(), v.end(), cmp);
	printf("sort finish: %d\n", int(clock() / CLOCKS_PER_SEC));
}

set<pair<int, pair<int, int > > > st;
int main(int argc,char*argv[]) {
	FILE* inp = fopen(argv[1], "r");
	FILE* ou = fopen(argv[2], "w");
	int k = stoi(argv[3]); //  保留列表里的最大不相交k个

	init(inp);
	fclose(inp);
	
	st.clear();
	for(auto it = v.begin(); it != v.end() && k>0; it++) {
		pair<int, pair<int, int > > cur = (*it).id;
		auto it2 = st.lower_bound(cur);
		
		bool chk = true;
		if(it2!=st.end() && intersect(cur, *it2)) chk = false;
		if(it2!=st.begin()) {
			it2--;
			if(intersect(cur, *it2)) chk = false;
		}			
		if(!chk) continue;
		
		k--;
		st.insert(cur);
		fprintf(ou, "%s\t%d\t%d\t%.4lf\t%.4lf\t%.4lf\n", (*it).chr_str().c_str(), (*it).st, (*it).ed, (*it).sc1, (*it).sc2, (*it).sc3);
	}
	fclose(ou);
	
	printf("done: %d\n", int(clock() / CLOCKS_PER_SEC));
	
	return 0;
}

    
