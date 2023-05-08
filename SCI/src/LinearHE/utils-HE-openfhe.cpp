/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "LinearHE/utils-HE-openfhe.h"
#include "seal/util/polyarithsmallmod.h"
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/bfvrns/bfvrns-ser.h"

using namespace std;
using namespace sci;
using namespace seal;
using namespace seal::util;
// using namespace lbcrypto;


// void generate_new_keys(int party, NetIO *io, int slot_count,
//                        lbcrypto::CCParams<lbcrypto::CryptoContextBFVRNS> &parameters,
//                        lbcrypto::CryptoContext<lbcrypto::DCRTPoly> *&cryptoContext, lbcrypto::KeyPair<lbcrypto::DCRTPoly> *&keyPair,
//                        bool verbose) {
//   cout << "Generating New Keys" << endl;

//   parameters.SetPlaintextModulus(4293918721);
//   parameters.SetMultiplicativeDepth(2);
//   // parameters.SetMaxRelinSkDeg(3);
//   parameters.SetSecurityLevel(HEStd_128_classic);

//   if (party == BOB) {
//     cryptoContext = GenCryptoContext(parameters);
//     cryptoContext->Enable(PKE);
//     cryptoContext->Enable(KEYSWITCH);
//     cryptoContext->Enable(LEVELEDSHE);
//     keyPair = cryptoContext->KeyGen();
//     cryptoContext->EvalMultKeyGen(keyPair.secretKey);
//     std::vector<int> v(32);
//     // std::iota(v.begin(), v.end(), 1);
//     // for (size_t i = 0; i < 14; i++)
//     //     v[i] = (int32_t) pow(2, i);
//     for (size_t i = 0; i < 32; i++)
//         v[i] = 128 * i;
//     // for (size_t i = 0; i < 128; i++)
//     //     v[i+78] = i;
//     cryptoContext->EvalRotateKeyGen(keyPair.secretKey, v);
//     cout << "Rotation IDs: " << v << endl;
//     std::for_each(v.begin(), v.end(), [](int& d) { d+=4096;});
//     cryptoContext->EvalRotateKeyGen(keyPair.secretKey, v);
//     cout << "Rotation IDs: " << v << endl;
//     // this->M = 4 * (this->cryptoContext->GetRingDimension());
//     cout << "Ring Dimension: " << cryptoContext->GetRingDimension();
//     cout << "Key generated " << keyPair.publicKey << endl;

//     stringstream os;
//     Serial::Serialize(keyPair, os, lbcrypto::SerType::BINARY);
//     // pub_key.save(os);
//     uint64_t kp_size = os.tellp();
//     Serial::Serialize(cryptoContext, os, lbcrypto::SerType::BINARY);

//     uint64_t cc_size = (uint64_t)os.tellp() - kp_size;

//     string key_context = os.str();
//     io->send_data(&kp_size, sizeof(uint64_t));
//     io->send_data(&cc_size, sizeof(uint64_t));
//     io->send_data(key_context.c_str(), kp_size + cc_size);

//   } else // party == ALICE
//   {
//     uint64_t kp_size;
//     uint64_t cc_size;
//     io->recv_data(&kp_size, sizeof(uint64_t));
//     io->recv_data(&cc_size, sizeof(uint64_t));
//     char *key_context = new char[kp_size + cc_size];
//     io->recv_data(key_context, kp_size + cc_size);
//     stringstream is_kp;

//     is_kp.write(key_context, kp_size);
//     Serial::Deserialize(keyPair, is_kp, lbcrypto::SerType::BINARY);
//     stringstream is_cc;
//     gal_keys_ = new GaloisKeys();
//     is_cc.write(key_context + kp_size, cc_size);
//     Serial::Deserialize(cryptoContext, is_cc, lbcrypto::SerType::BINARY);

//     delete[] key_context;
//   }
//   if (verbose)
//     cout << "Keys Generated (slot_count: " << slot_count << ")" << endl;
// }

void send_encrypted_vector_openfhe(NetIO *io, vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &ct_vec) {
  assert(ct_vec.size() > 0);
  stringstream os;
  uint64_t ct_size;
  for (size_t ct = 0; ct < ct_vec.size(); ct++) {
    // ct_vec[ct].save(os);
    lbcrypto::Serial::Serialize(ct_vec[ct], os, lbcrypto::SerType::BINARY);
    if (!ct)
      ct_size = os.tellp();
  }
  string ct_ser = os.str();
  io->send_data(&ct_size, sizeof(uint64_t));
  io->send_data(ct_ser.c_str(), ct_ser.size());
  // std::cout << "test ciphertext vector size: " << ct_ser.size() << " " << ct_vec.size() << std::endl;
}

void recv_encrypted_vector_openfhe(NetIO *io, vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &ct_vec) {
  assert(ct_vec.size() > 0);
  stringstream is;
  uint64_t ct_size;
  io->recv_data(&ct_size, sizeof(uint64_t));
  char *c_enc_result = new char[ct_size * ct_vec.size()];
  io->recv_data(c_enc_result, ct_size * ct_vec.size());
  for (size_t ct = 0; ct < ct_vec.size(); ct++) {
    is.write(c_enc_result + ct_size * ct, ct_size);
    // ct_vec[ct].unsafe_load(is);
    lbcrypto::Serial::Deserialize(ct_vec[ct], is, lbcrypto::SerType::BINARY);
  }
  delete[] c_enc_result;
}

void send_ciphertext_openfhe(NetIO *io, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct) {
  stringstream os;
  uint64_t ct_size;
  // ct.save(os);
  lbcrypto::Serial::Serialize(ct, os, lbcrypto::SerType::BINARY);
  ct_size = os.tellp();
  string ct_ser = os.str();
  io->send_data(&ct_size, sizeof(uint64_t));
  io->send_data(ct_ser.c_str(), ct_ser.size());
  // std::cout << "test ciphertext size: " << ct_ser.size() << std::endl;
}

void recv_ciphertext_openfhe(NetIO *io, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct) {
  stringstream is;
  uint64_t ct_size;
  io->recv_data(&ct_size, sizeof(uint64_t));
  char *c_enc_result = new char[ct_size];
  io->recv_data(c_enc_result, ct_size);
  is.write(c_enc_result, ct_size);
  lbcrypto::Serial::Deserialize(ct, is, lbcrypto::SerType::BINARY);
  // ct.unsafe_load(is);
  delete[] c_enc_result;
}

// void set_poly_coeffs_uniform_openfhe(
//     uint64_t *poly, uint32_t bitlen, shared_ptr<UniformRandomGenerator> random,
//     shared_ptr<const SEALContext::ContextData> &context_data) {
//   assert(bitlen < 128 && bitlen > 0);
//   auto &parms = context_data->parms();
//   auto &coeff_modulus = parms.coeff_modulus();
//   size_t coeff_count = parms.poly_modulus_degree();
//   size_t coeff_mod_count = coeff_modulus.size();
//   uint64_t bitlen_mask = (1ULL << (bitlen % 64)) - 1;
// 
//   RandomToStandardAdapter engine(random);
//   for (size_t i = 0; i < coeff_count; i++) {
//     if (bitlen < 64) {
//       uint64_t noise = (uint64_t(engine()) << 32) | engine();
//       noise &= bitlen_mask;
//       for (size_t j = 0; j < coeff_mod_count; j++) {
//         poly[i + (j * coeff_count)] =
//             barrett_reduce_63(noise, coeff_modulus[j]);
//       }
//     } else {
//       uint64_t noise[2]; // LSB || MSB
//       for (int j = 0; j < 2; j++) {
//         noise[0] = (uint64_t(engine()) << 32) | engine();
//         noise[1] = (uint64_t(engine()) << 32) | engine();
//       }
//       noise[1] &= bitlen_mask;
//       for (size_t j = 0; j < coeff_mod_count; j++) {
//         poly[i + (j * coeff_count)] =
//             barrett_reduce_128(noise, coeff_modulus[j]);
//       }
//     }
//   }
// }
// 
// void flood_ciphertext_openfhe(Ciphertext &ct,
//                       shared_ptr<const SEALContext::ContextData> &context_data,
//                       uint32_t noise_len, MemoryPoolHandle pool) {
// 
//   auto &parms = context_data->parms();
//   auto &coeff_modulus = parms.coeff_modulus();
//   size_t coeff_count = parms.poly_modulus_degree();
//   size_t coeff_mod_count = coeff_modulus.size();
// 
//   auto noise(allocate_poly(coeff_count, coeff_mod_count, pool));
//   shared_ptr<UniformRandomGenerator> random(parms.random_generator()->create());
// 
//   set_poly_coeffs_uniform_openfhe(noise.get(), noise_len, random, context_data);
//   for (size_t i = 0; i < coeff_mod_count; i++) {
//     add_poly_poly_coeffmod(noise.get() + (i * coeff_count),
//                            ct.data() + (i * coeff_count), coeff_count,
//                            coeff_modulus[i], ct.data() + (i * coeff_count));
//   }
// 
//   set_poly_coeffs_uniform_openfhe(noise.get(), noise_len, random, context_data);
//   for (size_t i = 0; i < coeff_mod_count; i++) {
//     add_poly_poly_coeffmod(noise.get() + (i * coeff_count),
//                            ct.data(1) + (i * coeff_count), coeff_count,
//                            coeff_modulus[i], ct.data(1) + (i * coeff_count));
//   }
// }
