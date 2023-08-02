/*
Original Author: ryanleh
Modified Work Copyright (c) 2020 Microsoft Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Modified by Deevashwer Rathee
*/

#ifndef BERTFC_FIELD_H__
#define BERTFC_FIELD_H__

#include "utils-HE.h"
#include <fstream>

using namespace std;
using namespace sci;
using namespace seal;

struct FCMetadata {
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

class PruneLin1Field {
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

  PruneLin1Field(int party, NetIO *io);

  ~PruneLin1Field();

  void configure();

  vector<vector<Plaintext>> generate_rotation_masks(const FCMetadata &data);
  vector<Plaintext> generate_cross_packing_masks(const FCMetadata &data);

  vector<Ciphertext> rotation_by_one_depth3(const FCMetadata &data, const Ciphertext &ct, int k);

  vector<Ciphertext> bert_efficient_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data);

  vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> bert_cross_packing_single_matrix(const vector<vector<vector<uint64_t>>> &weights, const FCMetadata &data);

  vector<vector<Plaintext>> bert_cross_packing_bias(const vector<vector<uint64_t>> &bias, const FCMetadata &data);

  Ciphertext bert_efficient_preprocess_noise(const uint64_t *secret_share, const FCMetadata &data);

  void bert_cipher_plain_bsgs(const vector<Ciphertext> &cts, const vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &wq_pack, const vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &wk_pack, const vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &wv_pack, const vector<vector<Plaintext>> &bq_pack, const vector<vector<Plaintext>> &bk_pack, const vector<vector<Plaintext>> &bv_pack, const FCMetadata &data, vector<Ciphertext> &result);

  void bert_cipher_cipher_cross_packing(const FCMetadata &data, const vector<Ciphertext> &Cipher_plain_result, const vector<Plaintext> &cross_masks, vector<Ciphertext> &results);

  uint64_t* bert_cross_packing_postprocess(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing=true);

  uint64_t* bert_postprocess_V(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing=true);

  // vector<uint64_t> ideal_functionality(uint64_t *vec, uint64_t **matrix);

  vector<Ciphertext> preprocess_softmax_s1(const uint64_t *matrix, const FCMetadata &data);

  vector<vector<vector<Plaintext>>> preprocess_softmax_s2(const uint64_t *matrix, const FCMetadata &data, vector<vector<vector<uint64_t>>> &mask);

  vector<vector<vector<uint64_t>>> softmax_mask(const FCMetadata &data);

  void bert_softmax_V(vector<Ciphertext> &softmax_s1, vector<vector<vector<Plaintext>>> &softmax_s2, vector<Ciphertext> &V, vector<vector<vector<Plaintext>>> &R, const FCMetadata &data, vector<Ciphertext> &result);

  vector<vector<vector<Plaintext>>> bert_softmax_v_packing_single_matrix(const vector<vector<vector<uint64_t>>> &weights, const FCMetadata &data);

  vector<vector<vector<Plaintext>>> preprocess_softmax_v_r(const uint64_t *matrix, const FCMetadata &data);

  uint64_t* client_S1_V_R(const uint64_t *softmax_s1, vector<Ciphertext> &V, const FCMetadata &data);

  vector<Plaintext> preprocess_softmax_s2_ct_ct(const uint64_t *matrix, const FCMetadata &data);

  void bert_softmax_V_ct_ct(vector<vector<vector<Ciphertext>>> &softmax_s2, vector<Ciphertext> &V, const FCMetadata &data, vector<Ciphertext> &result);
  vector<vector<vector<Ciphertext>>> preprocess_softmax_s1_ct_ct(const vector<Ciphertext> &matrix, const FCMetadata &data, vector<vector<Plaintext>> &mask);
  vector<vector<Plaintext>> softmax_mask_ct_ct(const FCMetadata &data);

  void print_noise_budget_vec(vector<Ciphertext> v);

  void print_ct(Ciphertext &ct, int len);
  void print_pt(Plaintext &pt, int len);
  void saveMatrix(const std::string& filename, uint64_t* matrix, size_t rows, size_t cols);

  void matrix_multiplication(int32_t input_dim, int32_t common_dim,
                            int32_t output_dim,
                            vector<vector<uint64_t>> &A, 
                            vector<vector<vector<uint64_t>>> &B1, 
                            vector<vector<vector<uint64_t>>> &B2, 
                            vector<vector<vector<uint64_t>>> &B3, 
                            vector<vector<uint64_t>> &Bias1, 
                            vector<vector<uint64_t>> &Bias2, 
                            vector<vector<uint64_t>> &Bias3, 
                            vector<vector<uint64_t>> &C, 
                            bool verify_output = false);

};
#endif
