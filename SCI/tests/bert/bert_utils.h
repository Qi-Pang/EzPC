#ifndef BERT_UTILS_H
#define BERT_UTILS_H

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include <vector>
#include <sstream>

using namespace std;

vector<vector<uint64_t>> read_data(const string& filename);
vector<uint64_t> read_bias(const string& filename, int output_dim) ;

vector<vector<vector<uint64_t>>> read_qkv_weights(const string& filename);
vector<vector<uint64_t>> read_qkv_bias(const string& filename);

#endif