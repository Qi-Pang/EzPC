/*
Original Author: ryanleh, Deevashwer Rathee
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

Modified by Qi Pang
*/

#include "LinearHE/bert-att-ct-pt.h"
#include <omp.h>

using namespace std;
using namespace sci;
using namespace seal;

#define HE_TIMING
// #define HE_DEBUG

void BEAttCtPt::print_noise_budget_vec(vector<Ciphertext> v) {
    cout << "Noise budget: ";
    for(int i = 0; i < v.size(); i++){
        cout << YELLOW << decryptor->invariant_noise_budget(v[i]) << " ";
    }
    cout << RESET << endl;
}

void BEAttCtPt::print_ct(Ciphertext &ct, int len){
    Plaintext pt;
    decryptor->decrypt(ct, pt);
    print_pt(pt, len);
}

void BEAttCtPt::print_pt(Plaintext &pt, int len) {
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
vector<Ciphertext> BEAttCtPt::bert_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data) {
    vector<int64_t> pod_matrix(data.slot_count, 0ULL);
    vector<Ciphertext> cts;
    for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
    {
        pod_matrix = vector<int64_t>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
        Ciphertext ct;
        Plaintext pt;
        encoder->encode(pod_matrix, pt);
        encryptor->encrypt(pt, ct);
        cts.push_back(ct);
    }
    return cts;
}

vector<Plaintext> BEAttCtPt::bert_cross_packing_bias(const uint64_t *matrix, const FCMetadata &data) {
    vector<Plaintext> cross_bias_packing(data.image_size * data.filter_w / data.slot_count);
    int matrix_pointer = 0;
    for (int packing_num = 0; packing_num < data.image_size * data.filter_w / data.slot_count; packing_num++) {
        vector<uint64_t> temp(data.slot_count, 0ULL);
        int right_flag = 0;
        int row = 0;
        while (row < data.slot_count) {
            for (int i = 0; i < data.image_size; i++) {
                temp[row + i] = matrix[matrix_pointer];
            }
            row += data.image_size;
            matrix_pointer++;
        }
        Plaintext pt;
        encoder->encode(temp, pt);
        cross_bias_packing[packing_num] = pt;
        temp.clear();
    }
    return cross_bias_packing;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> BEAttCtPt::bert_cross_packing_single_matrix(const uint64_t *const *matrix1, const uint64_t *const *matrix2, const FCMetadata &data) {
    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    vector<uint64_t> temp2;
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = data.filter_h / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag / 2; // should be 8

    int n1;
    int n2;
    if (data.slot_count == 4096) {
        n1 = 2;
        n2 = 8;
    }
    else {
        if (data.filter_h == 3072 && data.filter_w == 768) {
            n1 = 2;
            n2 = 16;
        }
        else if (data.filter_h == 768 && data.filter_w == 3072) {
            n1 = 8;
            n2 = 4;
        }
        else if (data.filter_h == 768 && data.filter_w == 768) {
            n1 = 4;
            n2 = 8;
        }
        else {
            assert (0);
        }
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back(matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back(matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            Plaintext pt;
                            encoder->encode(temp2, pt);
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix1.push_back(temp_matrix_diag);
        }
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back(matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back(matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                            Plaintext pt;
                            encoder->encode(temp2, pt);
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix2.push_back(temp_matrix_diag);
        }
    }
    return std::make_pair(weightMatrix1, weightMatrix2);
}

/* Generates a masking vector of random noise that will be applied to parts of
 * the ciphertext that contain leakage */
Ciphertext BEAttCtPt::bert_efficient_preprocess_noise(const uint64_t *secret_share, const FCMetadata &data) {
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


void BEAttCtPt::bert_cipher_plain_bsgs(const vector<Ciphertext> &cts, const vector<vector<Plaintext>> &enc_mat1, const vector<vector<Plaintext>> &enc_mat2, const vector<Plaintext> &enc_bias, const FCMetadata &data, vector<Ciphertext> &result) {

    vector<vector<Ciphertext>> rotatedIR(cts.size()); // cts.size() = 48
    int n1;
    int n2;
    if (data.slot_count == 4096) {
        n1 = 2;
        n2 = 8;
    }
    else {
        if (data.filter_h == 3072 && data.filter_w == 768) {
            n1 = 2;
            n2 = 16;
        }
        else if (data.filter_h == 768 && data.filter_w == 3072) {
            n1 = 8;
            n2 = 4;
        }
        else if (data.filter_h == 768 && data.filter_w == 768) {
            n1 = 4;
            n2 = 8;
        }
        else {
            assert (0);
        }
    }
    int num_diag = data.slot_count / data.image_size / 2;
    cout << "[Server] Online Start" << endl;
    auto t1 = high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < cts.size(); i++)
    {   
        vector<Ciphertext> tmp;
        tmp.push_back(cts[i]);

        for (int j = 1; j < n1; j++) {
            Ciphertext temp_rot;
            evaluator->rotate_rows(cts[i], (num_diag - j) * data.image_size, *gal_keys, temp_rot);
            tmp.push_back(temp_rot);
        }

        for (int j = 0; j < n1; j++) {
            Ciphertext temp_rot;
            evaluator->rotate_columns(tmp[j], *gal_keys, temp_rot);
            tmp.push_back(temp_rot);
        }

        rotatedIR[i] = tmp;
        tmp.clear();
    }

    #ifdef HE_DEBUG
        cout << "[Server] Budget after rotation" << endl;
        print_noise_budget_vec(rotatedIR[0]);
    #endif

    auto t2 = high_resolution_clock::now();
    auto ms_double = (t2 - t1)/1e+9;
    cout << "[Server] Online - rotation done " << ms_double.count() << endl;
    //compute matrix multiplication
    vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2));
    vector<vector<Ciphertext>> temp_results1(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2 * cts.size()));
    t1 = high_resolution_clock::now();

    // rotatedIR 48 x 16, enc_mat 64 x 48

    #pragma omp parallel for
    for (int k = 0; k < cts.size() * n2; k++) {
        int j = k / cts.size();
        int ct_i = k % cts.size();
        for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
            for (int i = 0; i < n1; i++) {
                Ciphertext ct1_l;
                Ciphertext ct1_r;
                evaluator->multiply_plain(rotatedIR[ct_i][i], enc_mat1[n1 * j + i + l * num_diag][ct_i], ct1_l);
                evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_mat2[n1 * j + i + l * num_diag][ct_i], ct1_r);
                if (i == 0)
                    evaluator->add(ct1_l, ct1_r, temp_results1[l][k]);
                else {
                    Ciphertext temp_add_ct;
                    evaluator->add(ct1_l, ct1_r, temp_add_ct);
                    evaluator->add_inplace(temp_results1[l][k], temp_add_ct);
                }
            }
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < n2; j++) {
        for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                if (ct_i == 0)
                    temp_results[l][j] = temp_results1[l][j * cts.size() + ct_i];
                else
                    evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * cts.size() + ct_i]);
            }
        }
        
    }

    #ifdef HE_DEBUG
        cout << "[Server] Budget after mult" << endl;
        print_noise_budget_vec(temp_results[0]);
    #endif

    #pragma omp parallel for
    for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
        Ciphertext ct;
        for (int k = 0; k < n2; k++) {
            if (k == 0)
                ct = temp_results[l][0];
            else {
                Ciphertext temp_rot_ct;
                evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *gal_keys, temp_rot_ct);
                evaluator->add_inplace(ct, temp_rot_ct);
            }
        }
        result[l] = ct;
        evaluator->add_plain_inplace(result[l], enc_bias[l]);
    }

    t2 = high_resolution_clock::now();
    ms_double = (t2 - t1)/1e+9;
    cout << "[Server] Online Done " << ms_double.count() << endl;

}


uint64_t* BEAttCtPt::bert_efficient_postprocess(vector<Ciphertext> &cts, const FCMetadata &data) {
    uint64_t *result = new uint64_t[data.image_size * data.filter_w];
    for (int i = 0; i < data.image_size * data.filter_w / data.slot_count; i++) {
        vector<int64_t> plain(data.slot_count, 0ULL);
        Plaintext pt;
        decryptor->decrypt(cts[i], pt);
        encoder->decode(pt, plain);

        #pragma omp parallel for
        for (int row = 0; row < data.slot_count; row++) {
            int j = row / data.image_size + i * data.slot_count / data.image_size;
            int k = row % data.image_size;
            result[k + j * data.image_size] = plain[row];
        }
    }
    return result;
}

BEAttCtPt::BEAttCtPt(int party, NetIO *io) {
    this->party = party;
    this->io = io;
    this->slot_count = 8192;

    generate_new_keys_ctpt(party, io, slot_count, context, encryptor, decryptor,
                    evaluator, encoder, gal_keys, relin_keys, zero);
}

BEAttCtPt::~BEAttCtPt() {
    free_keys(party, encryptor, decryptor, evaluator, encoder, gal_keys, zero);
}

void BEAttCtPt::configure() {
  data.slot_count = 8192;
  // Only works with a ciphertext that fits in a single ciphertext
  assert(data.slot_count >= data.image_size);

  data.filter_size = data.filter_h * data.filter_w;
  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / next_pow2(data.filter_w);
  // How many total ciphertexts we'll need
  data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
}

vector<uint64_t> BEAttCtPt::ideal_functionality(uint64_t *vec, uint64_t **matrix) {
  vector<uint64_t> result(data.filter_h, 0ULL);
  for (int row = 0; row < data.filter_h; row++) {
    for (int idx = 0; idx < data.filter_w; idx++) {
      uint64_t partial = vec[idx] * matrix[row][idx];
      result[row] = result[row] + partial;
    }
  }
  return result;
}

void BEAttCtPt::matrix_multiplication(int32_t input_dim, 
                                    int32_t common_dim, 
                                    int32_t output_dim, 
                                    vector<vector<uint64_t>> &A, 
                                    vector<vector<uint64_t>> &B, 
                                    vector<uint64_t> &Bias, 
                                    vector<vector<uint64_t>> &C, 
                                    bool verify_output) {

    data.filter_h = common_dim;
    data.filter_w = output_dim;
    data.image_size = input_dim;
    this->slot_count = 8192;
    configure();

    if (party == BOB) {  
        // Client
        vector<uint64_t> vec(common_dim * input_dim);
        for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                vec[j*input_dim + i] = A[i][j];

        auto cts = bert_preprocess_vec(vec, data);

        print_noise_budget_vec(cts);

        auto io_start = io->counter;
        send_encrypted_vector(io, cts);
        cout << "[Client] Input cts sent" << endl;
        cout << "[Client] Size of cts (Bytes): " << sizeof(Ciphertext) << " " << sizeof(Ciphertext) * cts.size() << endl;

        vector<Ciphertext> enc_result(data.image_size * data.filter_w / data.slot_count);
        recv_encrypted_vector(context, io, enc_result);
        cout << "[Client] Output cts received" << endl;
        cout << "[Client] size of cts (Bytes): " << io->counter - io_start << endl;

        print_noise_budget_vec(enc_result);
        // print_ct(enc_result[0], data.slot_count);

        auto HE_result = bert_efficient_postprocess(enc_result, data);

        // HACK
        for (int i = 64; i < 65; i++) {
            for (int j = 0; j < data.filter_w; j++)
                cout << ((int64_t) HE_result[i + j * 128] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
            cout << endl;
        }
        
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
        vector<Ciphertext> cts(data.image_size * data.filter_h / data.slot_count);
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
            matrix_mod_p1[i] = new uint64_t[output_dim / 2];
            matrix_mod_p2[i] = new uint64_t[output_dim / 2];

            matrix1[i] = new uint64_t[output_dim / 2];
            matrix2[i] = new uint64_t[output_dim / 2];

            for (int j = 0; j < output_dim / 2; j++) {
                matrix_mod_p1[i][j] = neg_mod((int64_t)B[i][j], (int64_t)prime_mod);
                matrix_mod_p2[i][j] = neg_mod((int64_t)B[i][j + output_dim / 2], (int64_t)prime_mod);
            }
        }
        for (int i = 0; i < output_dim; i++) {
            Bias[i] = neg_mod((int64_t)Bias[i], (int64_t)prime_mod);
        }

        PRG128 prg;
        uint64_t *secret_share = new uint64_t[input_dim*output_dim];
        prg.random_mod_p<uint64_t>(secret_share, input_dim*output_dim, prime_mod);
        // auto encoded_mat1 = bert_efficient_preprocess_matrix(matrix_mod_p1.data(), data);
        // auto encoded_mat2 = bert_efficient_preprocess_matrix(matrix_mod_p2.data(), data);

        // auto cross_mat = bert_cross_packing_matrix(matrix_mod_p1.data(), matrix_mod_p2.data(), data);
        auto cross_mat_single = bert_cross_packing_single_matrix(matrix_mod_p1.data(), matrix_mod_p2.data(), data);
        auto cross_bias_single = bert_cross_packing_bias(Bias.data(), data);

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

        // auto Cipher_plain_results = bert_efficient_online(cts, encoded_mat, encoded_mat, data, rotation_masks);
        // auto Cipher_plain_results = bert_cipher_plain(cts, cross_mat.first, cross_mat.second, data);
        vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w / data.slot_count);
        bert_cipher_plain_bsgs(cts, cross_mat_single.first, cross_mat_single.second, cross_bias_single, data, Cipher_plain_results);

        #ifdef HE_TIMING
        auto t2_cipher_plain = high_resolution_clock::now();
        interval = (t2_cipher_plain - t1_cipher_plain)/1e+9;
        cout << "[Server] Cipher-Plaintext Matmul takes " << interval.count() << "sec" << endl;
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