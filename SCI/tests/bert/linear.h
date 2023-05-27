#ifndef LINEAR_H__
#define LINEAR_H__

#include "LinearHE/utils-HE.h"
#include "FloatingPoint/fp-math.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>

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
	SEALContext *context;
	Encryptor *encryptor;
	Decryptor *decryptor;
	Evaluator *evaluator;
	BatchEncoder *encoder;
	GaloisKeys *gal_keys;
	RelinKeys *relin_keys;
	Ciphertext *zero;
	size_t slot_count;

	Linear();

	Linear(int party, NetIO *io);

	~Linear();

	void configure();

	void generate_new_keys();

	// vector<vector<Plaintext>> generate_rotation_masks();
  	// vector<Plaintext> generate_cipher_masks();
  	// vector<Plaintext> generate_packing_masks();
  	// vector<Plaintext> generate_depth3_masks();
 	// vector<Plaintext> generate_cross_packing_masks();

	// vector<Ciphertext> rotation_by_one_depth3(
	// 	const FCMetadata &data, const Ciphertext &ct, int k);

	// vector<Ciphertext>
	// bert_efficient_preprocess_vec(
	// 	vector<uint64_t> &input,
	// 	const FCMetadata &data);

	// pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
	// bert_cross_packing_matrix(
	// 	const uint64_t *const *matrix1,
	// 	const uint64_t *const *matrix2,
	// 	const FCMetadata &data);

	// pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
	// bert_cross_packing_single_matrix(
	// 	const uint64_t *const *matrix1,
	// 	const uint64_t *const *matrix2,
	// 	const FCMetadata &data);

	// void bert_cipher_plain_bsgs(
	// 	const vector<Ciphertext> &cts,
	// 	const vector<pair<vector<vector<Plaintext>>,
	// 		vector<vector<Plaintext>>>> &cross_mats,
	// 	const vector<pair<vector<vector<Plaintext>>,
	// 		vector<vector<Plaintext>>>> &cross_mats_single,
	// 	const FCMetadata &data,
	// 	vector<Ciphertext> &result);

	// void bert_cipher_cipher_cross_packing(
	// 	const FCMetadata &data,
	// 	const vector<Ciphertext> &Cipher_plain_result,
	// 	const vector<Plaintext> &cross_masks,
	// 	vector<Ciphertext> &results);

	// uint64_t *bert_cross_packing_postprocess(
	// 	vector<Ciphertext> &cts,
	// 	const FCMetadata &data);

	// // vector<uint64_t> ideal_functionality(
	// // 	uint64_t *vec,
	// // 	uint64_t **matrix);

	// void print_noise_budget_vec(vector<Ciphertext> v);

	// void print_ct(Ciphertext &ct, int len);
	// void print_pt(Plaintext &pt, int len);

	// void matrix_multiplication(
	// 	int32_t input_dim,
	// 	int32_t common_dim,
	// 	int32_t output_dim,
	// 	vector<vector<uint64_t>> &A,
	// 	vector<vector<uint64_t>> &B1,
	// 	vector<vector<uint64_t>> &B2,
	// 	vector<vector<uint64_t>> &C,
	// 	bool verify_output = false);
};

#endif
