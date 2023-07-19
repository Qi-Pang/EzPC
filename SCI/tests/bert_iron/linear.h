#ifndef LINEAR_H__
#define LINEAR_H__

#include "he.h"
#include "FloatingPoint/fp-math.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <math.h>
#include "bert_utils.h"

#define PACKING_NUM 12

#define INPUT_DIM 128
#define COMMON_DIM 768
#define OUTPUT_DIM 64
#define INTER_DIM 3072

#define ATTENTION_LAYERS 12

using namespace sci;
using namespace std;
using namespace seal;

#define MAX_THREADS 12
struct FCMetadata
{
	int slot_count;
	int32_t pack_num;
	int32_t inp_ct;
	// Filter is a matrix
	int32_t filter_h;
	int32_t filter_w;
	int32_t filter_size;
	// Image is a matrix
	int32_t image_size;
};

struct PreprocessParams_1{
    vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats;
	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats_single;
	vector<vector<Plaintext>> bias_packing;
    vector<Plaintext> cross_masks;
};

struct PreprocessParams_2{
    pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> cross_mat_single;
    vector<Plaintext> cross_bias_single;
};

class Linear
{
public:
	int party;
	NetIO *io;
	FCMetadata data;

	HE *he_4096;

	// Fix linking error
	uint64_t p_mod;

	FCMetadata data_lin1;
    FCMetadata data_lin2;
    FCMetadata data_lin3;
    FCMetadata data_lin4;

	// Attention
	vector<PreprocessParams_1> pp_1;
	vector<PreprocessParams_2> pp_2;
	vector<PreprocessParams_2> pp_3;
	vector<PreprocessParams_2> pp_4;

	// Layer Norm
    vector<vector<uint64_t>> w_ln_1;
    vector<vector<uint64_t>> b_ln_1;

    vector<vector<uint64_t>> w_ln_2;
    vector<vector<uint64_t>> b_ln_2;

	// Pooling
    vector<vector<uint64_t>> w_p;
    vector<uint64_t> b_p;

    // Classification
    vector<vector<uint64_t>> w_c;
    vector<uint64_t> b_c;

	Linear();

	Linear(int party, NetIO *io);

	~Linear();

	void concat( 
		uint64_t* input,
		uint64_t* output,
		int n,
		int dim1,
		int dim2);

};

#endif
