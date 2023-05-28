#ifndef LINEAR_H__
#define LINEAR_H__

#include "he.h"
#include "FloatingPoint/fp-math.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>

#define INPUT_DIM 128
#define COMMON_DIM 768
#define OUTPUT_DIM 64

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

class Linear
{
public:
	int party;
	NetIO *io;
	FCMetadata data;

	HE *he_8192;
	HE *he_4096;

	Linear();

	Linear(int party, NetIO *io);

	~Linear();

	void configure();

	void generate_new_keys();

	vector<Ciphertext> linear_1(
		HE* he,
		vector<Ciphertext> input_cts, 
		vector<vector<vector<uint64_t>>> w_q,
		vector<vector<vector<uint64_t>>> w_k,
		vector<vector<vector<uint64_t>>> w_v,
		vector<vector<uint64_t>> b_q,
		vector<vector<uint64_t>> b_k,
		vector<vector<uint64_t>> b_v,
		const FCMetadata &data
		);
	
	// vector<Ciphertext> linear_2(
	// 	HE* he,
	// 	vector<Ciphertext> input_cts, 
	// 	vector<vector<uint64_t>> w_o,
	// 	vector<vector<uint64_t>> b_o,
	// 	const FCMetadata &data
	// 	);

	// vector<Ciphertext> linear_inter(
	// 	HE* he,
	// 	vector<Ciphertext> input_cts, 
	// 	vector<vector<uint64_t>> w_i,
	// 	vector<vector<uint64_t>> b_i,
	// 	const FCMetadata &data
	// 	);


 	vector<Plaintext> generate_cross_packing_masks(HE* he, const FCMetadata &data);

	vector<Ciphertext> rotation_by_one_depth3(
	HE* he,
	const FCMetadata &data, 
	const Ciphertext &ct, 
	int k);

	vector<Ciphertext>
	bert_efficient_preprocess_vec(
		HE* he,
		vector<uint64_t> &input,
		const FCMetadata &data);

	pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
	bert_cross_packing_matrix(
		HE* he,
		const uint64_t *const *matrix1,
		const uint64_t *const *matrix2,
		const FCMetadata &data);

	pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
	bert_cross_packing_single_matrix(
		HE* he,
		const uint64_t *const *matrix1,
		const uint64_t *const *matrix2,
		const FCMetadata &data);

	vector<Plaintext> bert_cross_packing_bias(
		HE* he,
		const uint64_t *matrix1, 
		const uint64_t *matrix2, 
		const uint64_t *matrix3, 
		const FCMetadata &data);

	void bert_cipher_plain_bsgs(
		HE* he,
		const vector<Ciphertext> &cts, 
		const vector<pair<vector<vector<Plaintext>>, 
		vector<vector<Plaintext>>>> &cross_mats, 
		const vector<vector<Plaintext>> &Bias, 
		const vector<pair<vector<vector<Plaintext>>, 
		vector<vector<Plaintext>>>> &cross_mats_single, 
		const FCMetadata &data, 
		vector<Ciphertext> &result);

	void bert_cipher_cipher_cross_packing(
		HE* he,
		const FCMetadata &data,
		const vector<Ciphertext> &Cipher_plain_result,
		const vector<Plaintext> &cross_masks,
		vector<Ciphertext> &results);

};

#endif
