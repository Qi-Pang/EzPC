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

#include "LinearHE/iron-seal.h"
#include "seal/util/polyarithsmallmod.h"
#include <omp.h>

using namespace std;
using namespace seal;
using namespace sci;

#define HE_TIMING
// #define HE_DEBUG

void IRONFC::print_noise_budget_vec(vector<Ciphertext> v) {
    cout << "Noise budget: ";
    for(int i = 0; i < v.size(); i++){
        cout << YELLOW << decryptor->invariant_noise_budget(v[i]) << " ";
    }
    cout << RESET << endl;
}

Plaintext IRONFC::encode_vector(const uint64_t *vec, const FCMetadata &data) {
    Plaintext pt;
    pt.resize(data.slot_count);
    assert(pt.data() != nullptr);
    seal::util::modulo_poly_coeffs(vec, data.slot_count, prime_mod, pt.data());
    return pt;
}

// vector<Ciphertext> IRONFC::removeUnusedCoeffs(Ciphertext &cts, const Meta &meta, double *density) {
    
// }

void IRONFC::print_ct(Ciphertext &ct, int len){
    Plaintext pt;
    decryptor->decrypt(ct, pt);
    print_pt(pt, len);
}

void IRONFC::print_pt(Plaintext &pt, int len) {
    vector<int64_t> dest(len, 0ULL);
    encoder->decode(pt, dest);
    cout << "Decode first 5 rows: ";
    int non_zero_count;
    for(int i = 0; i < 5; i++){
        for (int j = 0; j < 64; j++)
            cout << dest[i + j * 128] << " ";
        cout << endl;
        // if(dest[i] != 0){
        //     non_zero_count += 1;
        // }
    }
    // cout << "Non zero count: " << non_zero_count;
    cout << endl;
}

// column-wise packing
vector<Ciphertext> IRONFC::preprocess_vec(vector<uint64_t> &input, const FCMetadata &data) {
    int nw = 16;
    int kw = 2;
    vector<Ciphertext> cts;
    for (int i = 0; i < (data.filter_h / nw); i++) {
        vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
        for (int row_index = 0; row_index < data.image_size; row_index++) {
            for (int col_index = 0; col_index < nw; col_index++) {
                pod_matrix[row_index * nw * kw + (nw - 1) - col_index] = input[row_index + (i * nw + col_index) * data.image_size];
            }
        }
        Plaintext pt = encode_vector(pod_matrix.data(), data);
        Ciphertext ct;
        encryptor->encrypt(pt, ct);

        cts.push_back(ct);
    }
    return cts;
}

vector<vector<Plaintext>> IRONFC::preprocess_matrix(const uint64_t *const *matrix, const FCMetadata &data) {
    vector<vector<Plaintext>> weightMatrix;
    int nw = 16;
    int kw = 2;
    int sub_mat_row = data.filter_h / nw; // 48
    int sub_mat_col = data.filter_w / kw; // 32
    for (int sub_row_ind = 0; sub_row_ind < sub_mat_row; sub_row_ind++) {
        vector<Plaintext> temp;
        for (int sub_col_ind = 0; sub_col_ind < sub_mat_col; sub_col_ind++) {
            vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
            for (int i = 0; i < nw; i++) {
                for (int j = 0; j < kw; j++) {
                    pod_matrix[j * nw + i] = matrix[i + sub_row_ind * nw][j + sub_col_ind * kw];
                }
            }
            Plaintext pt = encode_vector(pod_matrix.data(), data);
            temp.push_back(pt);
        }
        weightMatrix.push_back(temp);
        temp.clear();
    }
    return weightMatrix;
}

/* Generates a masking vector of random noise that will be applied to parts of
 * the ciphertext that contain leakage */
Ciphertext IRONFC::preprocess_noise(const uint64_t *secret_share, const FCMetadata &data) {
  // Sample randomness into vector
  vector<int64_t> noise(data.slot_count, 0ULL);

  for (int i = 0; i < data.slot_count; i++)
    noise[i] = secret_share[i];

  Plaintext pt;
  encoder->encode(noise, pt);
  Ciphertext enc_noise;
  encryptor->encrypt(pt, enc_noise);

  return enc_noise;
}

vector<Ciphertext> IRONFC::bert_cipher_plain(const vector<Ciphertext> &cts, const vector<vector<vector<Plaintext>>> &enc_mats1, const vector<vector<vector<Plaintext>>> &enc_mats2, const vector<vector<vector<Plaintext>>> &enc_mats3, const FCMetadata &data) {
    cout << "[Server] Online Start" << endl;
    int nw = 16;
    int kw = 2;

    vector<Ciphertext> result(data.filter_w / kw * 3 * 12);

    for (int packing_index = 0; packing_index < 12; packing_index++) {
        vector<vector<Plaintext>> enc_mat1 = enc_mats1[packing_index];
        vector<vector<Plaintext>> enc_mat2 = enc_mats2[packing_index];
        vector<vector<Plaintext>> enc_mat3 = enc_mats3[packing_index];

        #pragma omp parallel for
        for (int j = 0; j < data.filter_w / kw; j++) {
            for (int i = 0; i < data.filter_h / nw; i++) {
                Ciphertext temp_ct1;
                Ciphertext temp_ct2;
                Ciphertext temp_ct3;
                evaluator->multiply_plain(cts[i], enc_mat1[i][j], temp_ct1);
                evaluator->multiply_plain(cts[i], enc_mat2[i][j], temp_ct2);
                evaluator->multiply_plain(cts[i], enc_mat3[i][j], temp_ct3);
                if (i == 0) {
                    result[j + data.filter_w / kw * 3 * packing_index] = temp_ct1;
                    result[j + data.filter_w / kw + data.filter_w / kw * 3 * packing_index] = temp_ct2;
                    result[j + data.filter_w / kw * 2 + data.filter_w / kw * 3 * packing_index] = temp_ct3;
                }
                else {
                    evaluator->add_inplace(result[j + data.filter_w / kw * 3 * packing_index], temp_ct1);
                    evaluator->add_inplace(result[j + data.filter_w / kw + data.filter_w / kw * 3 * packing_index], temp_ct2);
                    evaluator->add_inplace(result[j + data.filter_w / kw * 2 + data.filter_w / kw * 3 * packing_index], temp_ct3);
                }
            }
        }
    }

    int L = result[0].coeff_modulus_size();
    vector<int> used_indices;
    for (int i = 0; i < data.image_size; i++) {
        for (int j = 0; j < kw; j++) {
            used_indices.push_back(i * nw * kw + (j + 1) * nw - 1);
        }
    }
    std::sort(used_indices.begin(), used_indices.end());

    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < data.slot_count; j++) {
            if (std::binary_search(used_indices.cbegin(), used_indices.cend(), j))
                continue;
            auto rns_ptr = result[i].data(0);
            for (int k = 0; k < L; k++) {
                rns_ptr[j] = 0;
                rns_ptr += data.slot_count;
            }
        }
    }
    return result;
}

uint64_t* IRONFC::bert_postprocess(vector<Ciphertext> &cts, const FCMetadata &data) {
    uint64_t *result = new uint64_t[data.image_size * data.filter_w * 3];
    int nw = 16;
    int kw = 2;
    
    for (int ct_ind = 0; ct_ind < cts.size() / 3 / 12; ct_ind++) {
        Plaintext pt;
        decryptor->decrypt(cts[ct_ind], pt);
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < kw; j++) {
                result[i + (j + ct_ind * kw) * data.image_size] = pt[i * nw * kw + (j + 1) * nw - 1];
            }
        }
    }

    for (int ct_ind = 0; ct_ind < cts.size() / 3 / 12; ct_ind++) {
        Plaintext pt;
        decryptor->decrypt(cts[ct_ind + cts.size() / 3 / 12], pt);
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < kw; j++) {
                result[i + (j + ct_ind * kw) * data.image_size + data.image_size * data.filter_w] = pt[i * nw * kw + (j + 1) * nw - 1];
            }
        }
    }

    for (int ct_ind = 0; ct_ind < cts.size() / 3 / 12; ct_ind++) {
        Plaintext pt;
        decryptor->decrypt(cts[ct_ind + cts.size() / 3 / 12 * 2], pt);
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < kw; j++) {
                result[i + (j + ct_ind * kw) * data.image_size + data.image_size * data.filter_w * 2] = pt[i * nw * kw + (j + 1) * nw - 1];
            }
        }
    }
    return result;
}

IRONFC::IRONFC(int party, NetIO *io) {
    this->party = party;
    this->io = io;
    this->slot_count = 4096;

    generate_new_keys_iron(party, io, slot_count, context, encryptor, decryptor,
                    evaluator, encoder, gal_keys, relin_keys, zero, true);
}

IRONFC::~IRONFC() {
    free_keys(party, encryptor, decryptor, evaluator, encoder, gal_keys, zero);
}

void IRONFC::configure() {
  data.slot_count = 4096;
  // Only works with a ciphertext that fits in a single ciphertext
  assert(data.slot_count >= data.image_size);

  data.filter_size = data.filter_h * data.filter_w;
  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / next_pow2(data.filter_w);
  // How many total ciphertexts we'll need
  data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
}

vector<uint64_t> IRONFC::ideal_functionality(uint64_t *vec, uint64_t **matrix) {
  vector<uint64_t> result(data.filter_h, 0ULL);
  for (int row = 0; row < data.filter_h; row++) {
    for (int idx = 0; idx < data.filter_w; idx++) {
      uint64_t partial = vec[idx] * matrix[row][idx];
      result[row] = result[row] + partial;
    }
  }
  return result;
}

void IRONFC::matrix_multiplication(int32_t input_dim, 
                                      int32_t common_dim, 
                                      int32_t output_dim, 
                                      vector<vector<uint64_t>> &A, 
                                      vector<vector<uint64_t>> &B1, 
                                      vector<vector<uint64_t>> &B2, 
                                      vector<vector<uint64_t>> &C, 
                                      bool verify_output) {

    data.filter_h = common_dim;
    data.filter_w = output_dim;
    data.image_size = input_dim;
    this->slot_count = 4096;
    configure();

    
    if (party == BOB) {  
        // Client
        vector<uint64_t> vec(common_dim * input_dim);
        for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                vec[j*input_dim + i] = A[i][j];

        #ifdef HE_TIMING
        auto t1_enc = high_resolution_clock::now();
        #endif

        auto cts = preprocess_vec(vec, data);

        print_noise_budget_vec(cts);

        #ifdef HE_TIMING
        auto t2_enc = high_resolution_clock::now();
        auto interval = (t2_enc - t1_enc)/1e+9;
        cout << "[Client] Encrypting takes " << interval.count() << "sec" << endl;
        #endif

        auto io_start = io->counter;
        send_encrypted_vector(io, cts);
        cout << "[Client] Input cts sent" << endl;
        cout << "[Client] Size of cts (Bytes): " << sizeof(Ciphertext) << " " << sizeof(Ciphertext) * cts.size() << endl;

        vector<Ciphertext> enc_result(32 * 3 * 12);
        recv_encrypted_vector(context, io, enc_result);
        cout << "[Client] Output cts received" << endl;
        cout << "[Client] size of cts (Bytes): " << io->counter - io_start << endl;

        // print_noise_budget_vec(enc_result);
        // print_ct(enc_result[0], data.slot_count);

        auto HE_result = bert_postprocess(enc_result, data);

        // HACK
        for (int i = 0; i < 3; i++) {
            for (int j = 64; j < 128; j++)
                cout << ((int64_t) HE_result[i + j * 128] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
            cout << endl;
        }

        #ifdef HE_DEBUG
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 64; j++)
                cout << ((int64_t) HE_result[i + j * 128] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
            cout << endl;
        }
        #endif
        
        // for (int i = 0; i < num_rows; i++) {
        //   C[i][0] = HE_result[i];
        // }
        // if (verify_output)
        //   verify(&vec, nullptr, C);

        delete[] HE_result;
    } else {
        // Server
        #ifdef HE_TIMING
        auto t1_total = high_resolution_clock::now();
        #endif 

        auto io_start = io->counter;
        vector<Ciphertext> cts(48);
        recv_encrypted_vector(this->context, io, cts);

        // vector<uint64_t> vec(common_dim);
        // for (int i = 0; i < common_dim; i++) {
        //     vec[i] = B[i][0];
        // }

        #ifdef HE_TIMING
        auto t1_preprocess = high_resolution_clock::now();
        #endif

        vector<uint64_t *> matrix_mod_p1(common_dim);
        vector<uint64_t *> matrix_mod_p2(common_dim);
        vector<uint64_t *> matrix1(common_dim);
        vector<uint64_t *> matrix2(common_dim);
        for (int i = 0; i < common_dim; i++) {
            matrix_mod_p1[i] = new uint64_t[output_dim];
            matrix_mod_p2[i] = new uint64_t[output_dim];
            matrix1[i] = new uint64_t[output_dim];
            matrix2[i] = new uint64_t[output_dim];
            for (int j = 0; j < output_dim; j++) {
                matrix_mod_p1[i][j] = neg_mod((int64_t)B1[i][j], (int64_t)prime_mod);
                matrix_mod_p2[i][j] = neg_mod((int64_t)B2[i][j], (int64_t)prime_mod);
                int64_t val = (int64_t)B1[i][j];
                if (val > int64_t(prime_mod / 2)) {
                    val = val - prime_mod;
                }
                matrix1[i][j] = val;
                val = (int64_t)B2[i][j];
                if (val > int64_t(prime_mod / 2)) {
                    val = val - prime_mod;
                }
                matrix2[i][j] = val;
            }
        }

        PRG128 prg;
        uint64_t *secret_share = new uint64_t[input_dim*output_dim];
        prg.random_mod_p<uint64_t>(secret_share, input_dim*output_dim, prime_mod);

        vector<vector<vector<Plaintext>>> encoded_mats1(12);
        vector<vector<vector<Plaintext>>> encoded_mats2(12);
        vector<vector<vector<Plaintext>>> encoded_mats3(12);
        for (int i = 0; i < 12; i++) {
            auto encoded_mat1 = preprocess_matrix(matrix_mod_p1.data(), data);
            auto encoded_mat2 = preprocess_matrix(matrix_mod_p2.data(), data);
            auto encoded_mat3 = preprocess_matrix(matrix_mod_p2.data(), data);
            encoded_mats1[i] = encoded_mat1;
            encoded_mats2[i] = encoded_mat2;
            encoded_mats3[i] = encoded_mat3;
        }
        
        // Ciphertext enc_noise = bert_efficient_preprocess_noise(secret_share, data, cryptoContext_, keyPair_);
        // cout << "[Server] Noise processed" << endl;
        #ifdef HE_TIMING
        auto t2_preprocess = high_resolution_clock::now();
        auto interval = (t2_preprocess - t1_preprocess)/1e+9;
        cout << "[Server] Preprocessing takes " << interval.count() << "sec" << endl;
        #endif

        #ifdef HE_DEBUG
            print_noise_budget_vec(cts);
        #endif

        #ifdef HE_TIMING
        auto t1_cipher_plain = high_resolution_clock::now();
        #endif 

        auto Cipher_plain_results = bert_cipher_plain(cts, encoded_mats1, encoded_mats2, encoded_mats3, data);

        #ifdef HE_TIMING
        auto t2_cipher_plain = high_resolution_clock::now();
        interval = (t2_cipher_plain - t1_cipher_plain)/1e+9;
        cout << "[Server] Cipher-Plaintext Matmul takes " << interval.count() << "sec" << endl;

        auto t1_cipher_cipher = high_resolution_clock::now();
        #endif 

        send_encrypted_vector(io, Cipher_plain_results);

        cout << "[Server] Result sent" << endl;
        cout << "[Server] size of result (Bytes): " << io->counter - io_start << endl;

        for (int i = 0; i < common_dim; i++) {
            delete[] matrix_mod_p1[i];
            delete[] matrix_mod_p2[i];
            delete[] matrix1[i];
            delete[] matrix2[i];
        }
        delete[] secret_share;

        #ifdef HE_TIMING
        auto t2_total = high_resolution_clock::now();
        interval = (t2_total - t1_total)/1e+9;
        cout << "[Server] Total Time " << interval.count() << "sec" << endl;
        #endif 
    }
}