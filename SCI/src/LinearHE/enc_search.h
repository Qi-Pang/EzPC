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

class SearchCTPT {
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

  SearchCTPT(int party, NetIO *io);

  ~SearchCTPT();

  void configure();

  void print_noise_budget_vec(vector<Ciphertext> v);

  void print_ct(Ciphertext &ct, int len);
  void print_pt(Plaintext &pt, int len);
  
  vector<Ciphertext> search_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data);

  uint64_t* search_postprocess(vector<Ciphertext> &cts, const FCMetadata &data);

  vector<Plaintext> search_packing_db(const uint64_t *const *matrix1, const FCMetadata &data);

  vector<Plaintext> search_packing_mask(const FCMetadata &data);

  void search_inner_prod(const vector<Ciphertext> &cts, const vector<Plaintext> &dbembedding, const vector<Plaintext> &mask, const FCMetadata &data, vector<Ciphertext> &result);

  void search(int32_t input_dim, int32_t common_dim,
                             int32_t output_dim,
                             vector<vector<uint64_t>> &A,
                             vector<vector<uint64_t>> &B1,
                             bool verify_output = false);

              
  




};
#endif
