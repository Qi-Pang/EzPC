#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>


using namespace std;

void linear_1_plain(
    uint64_t* input,
    vector<vector<vector<uint64_t>>> w_q,
    vector<vector<vector<uint64_t>>> w_k,
    vector<vector<vector<uint64_t>>> w_v,
    vector<vector<uint64_t>> b_q,
    vector<vector<uint64_t>> b_k,
    vector<vector<uint64_t>> b_v,
    uint64_t* q,
    uint64_t* k,
    uint64_t* v
);

void linear_2_plain(
    uint64_t* input,
    vector<vector<uint64_t>> w,
    vector<uint64_t> b,
    int dim1,
    int dim2,
    int dim3,
    uint64_t* output
);