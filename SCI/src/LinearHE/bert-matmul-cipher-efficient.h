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

#include "LinearHE/utils-HE-openfhe.h"

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

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> bert_efficient_preprocess_vec(std::vector<std::vector<uint64_t>> &input, const FCMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext, lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keyPair);

std::vector<std::vector<lbcrypto::Plaintext>> bert_efficient_preprocess_matrix(const uint64_t *const *matrix, const FCMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext);

std::vector<std::vector<lbcrypto::Plaintext>> generate_rotation_masks(const FCMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext);

std::vector<lbcrypto::Plaintext> generate_cipher_masks(const FCMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext);

std::vector<lbcrypto::Plaintext> generate_packing_masks(const FCMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> rotation_by_one(const FCMetadata &data, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct, int k, std::vector<std::vector<lbcrypto::Plaintext>> rotation_masks, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> hoisted_rotation_by_one(const FCMetadata &data, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct, int k, std::vector<std::vector<lbcrypto::Plaintext>> rotation_masks, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext, std::shared_ptr<std::vector<lbcrypto::DCRTPoly>> &rotation_precompute);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> bert_efficient_preprocess_noise(const uint64_t *secret_share, const FCMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext, lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keyPair);

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> bert_efficient_online(std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &cts, std::vector<std::vector<lbcrypto::Plaintext>> &enc_mat1, std::vector<std::vector<lbcrypto::Plaintext>> &enc_mat2, const FCMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext, lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keyPair, std::vector<std::vector<lbcrypto::Plaintext>> & rotation_masks);

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> bert_efficient_cipher(const FCMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext, lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keyPair, std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> Cipher_plain_result, std::vector<std::vector<lbcrypto::Plaintext>>& rotation_masks, std::vector<lbcrypto::Plaintext>& cipher_masks);

uint64_t *bert_efficient_postprocess(std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &result, const FCMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cryptoContext, lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keyPair);

class BERTEFFICIENTFCFieldOPENFHE {
public:
  int party;
  sci::NetIO *io;
  FCMetadata data;

    lbcrypto::CCParams<lbcrypto::CryptoContextBFVRNS> parameters;
    // lbcrypto::CCParams<lbcrypto::CryptoContextBGVRNS> parameters;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keyPair;
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cryptoContext;
    uint32_t M;
    std::vector<lbcrypto::Plaintext> columnMasks;
    lbcrypto::Plaintext gazelleMask;
  size_t slot_count;

  BERTEFFICIENTFCFieldOPENFHE(int party, sci::NetIO *io);

  ~BERTEFFICIENTFCFieldOPENFHE();

  void configure();

  std::vector<uint64_t> ideal_functionality(uint64_t *vec, uint64_t **matrix);

  void matrix_multiplication(int32_t input_dim, int32_t common_dim,
                             int32_t output_dim,
                             std::vector<std::vector<uint64_t>> &A,
                             std::vector<std::vector<uint64_t>> &B,
                             std::vector<std::vector<uint64_t>> &C,
                             bool verify_output = false, bool verbose = false);

//   void verify(std::vector<uint64_t> *vec, std::vector<uint64_t *> *matrix,
//               std::vector<std::vector<uint64_t>> &C);
};
#endif
