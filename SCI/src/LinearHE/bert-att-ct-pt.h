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

class BEAttCtPt {
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

  BEAttCtPt(int party, NetIO *io);

  ~BEAttCtPt();

  void configure();

  vector<Ciphertext> bert_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data);

  pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> bert_cross_packing_single_matrix(const uint64_t *const *matrix1, const uint64_t *const *matrix2, const FCMetadata &data);

  vector<Plaintext> bert_cross_packing_bias(const uint64_t *matrix, const FCMetadata &data);

  Ciphertext bert_efficient_preprocess_noise(const uint64_t *secret_share, const FCMetadata &data);

  void bert_cipher_plain_bsgs(const vector<Ciphertext> &cts, const vector<vector<Plaintext>> &enc_mat1, const vector<vector<Plaintext>> &enc_mat2, const vector<Plaintext> &enc_bias, const FCMetadata &data, vector<Ciphertext> &result);

  uint64_t* bert_efficient_postprocess(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing=true);

  // uint64_t* bert_cross_packing_postprocess(vector<Ciphertext> &cts, const FCMetadata &data);

  vector<uint64_t> ideal_functionality(uint64_t *vec, uint64_t **matrix);

  void print_noise_budget_vec(vector<Ciphertext> v);

  void print_ct(Ciphertext &ct, int len);
  void print_pt(Plaintext &pt, int len);
  void saveMatrix(const std::string& filename, uint64_t* matrix, size_t rows, size_t cols);

  void matrix_multiplication(int32_t input_dim, int32_t common_dim,
                             int32_t output_dim,
                             vector<vector<uint64_t>> &A,
                             vector<vector<uint64_t>> &B,
                             vector<uint64_t> &Bias,
                             vector<vector<uint64_t>> &C,
                             bool verify_output = false);
};
#endif