#ifndef LINEAR_H__
#define LINEAR_H__

#include "he.h"
#include "FloatingPoint/fp-math.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>

#define PACKING_NUM 12

#define INPUT_DIM 128
#define COMMON_DIM 768
#define OUTPUT_DIM 64
#define INTER_DIM 3072

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

	// Fix linking error
	uint64_t p_mod;

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
	
	vector<Ciphertext> linear_2(
		HE* he,
		int32_t input_dim, 
		int32_t common_dim, 
		int32_t output_dim,
		vector<Ciphertext> input_cts, 
		vector<vector<uint64_t>> w,
		vector<uint64_t> b,
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

	// concat on dim1
	// output: dim2 x (dim1xdim3)
	vector<vector<uint64_t>> concat(
		uint64_t* att,
		int dim1,
		int dim2,
		int dim3);

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
	
	
	uint64_t* bert_cross_packing_postprocess(
		HE* he,
		vector<Ciphertext> &cts, 
		const FCMetadata &data);
	
	void plain_cross_packing_postprocess(
		uint64_t* input, 
		uint64_t * output,
		bool col_packing,
		const FCMetadata &data);
	
	void plain_cross_packing_postprocess_v(
		uint64_t* input, 
		uint64_t * output,
		bool col_packing,
		const FCMetadata &data);
	
	void plain_col_packing_preprocess(
		uint64_t* input, 
		uint64_t * output,
		uint64_t prime_mod,
		int common_dim,
		int input_dim);
	
	void plain_col_packing_preprocess_vec(
		vector<vector<uint64_t>> input, 
		uint64_t * output,
		uint64_t prime_mod,
		int common_dim,
		int input_dim);
	
	void plain_col_packing_postprocess(
		uint64_t* input, 
		uint64_t * output,
		bool col_packing,
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
	
	pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
	bert_cross_packing_single_matrix_2(
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
	
	vector<Plaintext> bert_cross_packing_bias_2(
		HE* he,
		const uint64_t *matrix, 
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

	void bert_cipher_plain_bsgs_2(
		HE* he,
		const vector<Ciphertext> &cts, 
		const vector<vector<Plaintext>> &enc_mat1, 
		const vector<vector<Plaintext>> &enc_mat2, 
		const vector<Plaintext> &enc_bias, 
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
