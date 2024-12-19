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

#include "LinearHE/enc_search.h"
#include <omp.h>
#include <fstream>

using namespace std;
using namespace sci;
using namespace seal;

#define HE_TIMING
// #define HE_DEBUG

void SearchCTPT::print_noise_budget_vec(vector<Ciphertext> v) {
    cout << "Noise budget: ";
    for(int i = 0; i < v.size(); i++){
        cout << YELLOW << decryptor->invariant_noise_budget(v[i]) << " ";
    }
    cout << RESET << endl;
}

void SearchCTPT::print_ct(Ciphertext &ct, int len){
    Plaintext pt;
    decryptor->decrypt(ct, pt);
    print_pt(pt, len);
}

void SearchCTPT::print_pt(Plaintext &pt, int len) {
    vector<int64_t> dest(len, 0ULL);
    encoder->decode(pt, dest);
    for(int i = 0; i < len; i++){
        cout << dest[i] << " ";
    }
    cout << endl;
}

vector<Ciphertext> SearchCTPT::search_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data) {
    vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
    vector<Ciphertext> cts;
    // int num_cts = (data.image_size * data.filter_h) / data.slot_count;
    for (int i = 0; i < input.size(); i++)
    {
        pod_matrix[i] = input[i];
    }
    Ciphertext ct;
    Plaintext pt;
    encoder->encode(pod_matrix, pt);
    encryptor->encrypt(pt, ct);
    cts.push_back(ct);
    return cts;
}

vector<Plaintext> SearchCTPT::search_packing_db(const uint64_t *const *matrix1, const FCMetadata &data) {
    vector<Plaintext> dbembedding; // output_dim / 2

    vector<uint64_t> temp2;

    for (int i = 0; i < data.filter_w; i++) { // 16 * 1024
        for (int j = 0; j < data.filter_h; j++) { // 4096
            temp2.push_back(matrix1[i][j]);
            if (temp2.size() % data.slot_count == 0) {
                Plaintext pt;
                encoder->encode(temp2, pt);
                dbembedding.push_back(pt);
                temp2.clear();
            }
        }
    }

    return dbembedding;
}

vector<Plaintext> SearchCTPT::search_packing_mask(const FCMetadata &data) {
    vector<Plaintext> mask_packing; // output_dim / 2

    vector<uint64_t> temp2;

    for (int j = 0; j < data.slot_count / 2; j++) { // 512
        vector<uint64_t> temp2(data.slot_count, 0ULL);
        for (int k = 0; k < data.slot_count / data.filter_h; k++) {
            temp2[j + k * data.filter_h] = 1;
        }
        Plaintext pt;
        encoder->encode(temp2, pt);
        mask_packing.push_back(pt);
        temp2.clear();
    }

    return mask_packing;
}

void SearchCTPT::search_inner_prod(const vector<Ciphertext> &cts, const vector<Plaintext> &dbembedding, const vector<Plaintext> &mask, const FCMetadata &data, vector<Ciphertext> &result) {

    auto t1 = high_resolution_clock::now();

    cout << "[Server] Online Start" << endl;

    vector<Ciphertext> temp_result(data.filter_w * data.filter_h / data.slot_count);

    for (int i = 0; i < result.size(); i++) {
        result[i] = *zero;
    }

    for (int i = 0; i < temp_result.size(); i++) {
        temp_result[i] = *zero;
    }

    Ciphertext temp_dup;
    Ciphertext ct;
    ct = cts[0];
    evaluator->rotate_columns(ct, *gal_keys, temp_dup);
    evaluator->add_inplace(ct, temp_dup);

    #pragma omp parallel for
    for (int i = 0; i < dbembedding.size(); i++) // 12 * 1024 / 2
    {
        Ciphertext temp_mul;
        evaluator->multiply_plain(ct, dbembedding[i], temp_mul);

        temp_result[i] = temp_mul;

        // print_noise_budget_vec(temp_result);
        for (int j = 0; j < 12; j++)
        {
            Ciphertext temp_rot;
            evaluator->rotate_rows(temp_result[i], int(pow(2, j)), *gal_keys, temp_rot);
            evaluator->add_inplace(temp_result[i], temp_rot);
        }
        // int res_cipher_ind = i / (data.slot_count / 2);
        int mask_ind = i % (data.slot_count / 2);
        evaluator->multiply_plain(temp_result[i], mask[mask_ind], temp_mul);
        temp_result[i] = temp_mul;
        // evaluator->add_inplace(result[res_cipher_ind], temp_result[i]);
        // if (i > dbembedding.size() - 500)
        //     print_noise_budget_vec(result);
    }

    for (int i = 0; i < temp_result.size(); i++) {
        int res_cipher_ind = i / (data.slot_count / 2);
        evaluator->add_inplace(result[res_cipher_ind], temp_result[i]);
    }
}

uint64_t* SearchCTPT::search_postprocess(vector<Ciphertext> &cts, const FCMetadata &data) {
    uint64_t *result = new uint64_t[data.filter_w];
    int num_cts_first_2 = data.filter_w / data.slot_count;
    for (int i = 0; i < num_cts_first_2; i++) {
        vector<int64_t> plain(data.slot_count, 0ULL);
        Plaintext pt;
        decryptor->decrypt(cts[i], pt);
        encoder->decode(pt, plain);

        #pragma omp parallel for
        for (int row = 0; row < data.slot_count / 2; row++) {
            result[row * 2 + data.slot_count * i] = plain[row];
            result[row * 2 + data.slot_count * i + 1] = plain[row + data.slot_count / 2];
        }
    }
    return result;
}

SearchCTPT::SearchCTPT(int party, NetIO *io) {
    this->party = party;
    this->io = io;
    this->slot_count = 8192;

    generate_new_keys_search(party, io, slot_count, context, encryptor, decryptor,
                    evaluator, encoder, gal_keys, relin_keys, zero);
}

SearchCTPT::~SearchCTPT() {
    free_keys(party, encryptor, decryptor, evaluator, encoder, gal_keys, zero);
}

void SearchCTPT::configure() {
  data.slot_count = 8192;
  // Only works with a ciphertext that fits in a single ciphertext
//   assert(data.slot_count >= data.image_size);

  data.filter_size = data.filter_h * data.filter_w;
  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / next_pow2(data.filter_w);
  // How many total ciphertexts we'll need
  data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
}

void SearchCTPT::search(int32_t input_dim, 
                    int32_t common_dim, 
                    int32_t output_dim, 
                    vector<vector<uint64_t>> &A, 
                    vector<vector<uint64_t>> &B1, 
                    bool verify_output) {

    data.filter_h = common_dim; // 4096
    data.filter_w = output_dim; // 16 * 1024
    data.image_size = input_dim; // 1
    this->slot_count = 8192;
    configure();

    if (party == BOB) {  
        // Client
        vector<uint64_t> vec(common_dim * input_dim);
        for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                vec[j*input_dim + i] = neg_mod((int64_t)A[i][j], (int64_t)prime_mod);

        auto cts = search_preprocess_vec(vec, data);

        print_noise_budget_vec(cts);

        auto io_start = io->counter;
        send_encrypted_vector(io, cts);
        cout << "[Client] Input cts sent" << endl;
        cout << "[Client] Size of cts (Bytes): " << sizeof(Ciphertext) << " " << sizeof(Ciphertext) * cts.size() << endl;

        vector<Ciphertext> enc_result(data.filter_w / data.slot_count);
        recv_encrypted_vector(context, io, enc_result);
        cout << "[Client] Output cts received" << endl;
        cout << "[Client] size of cts (Bytes): " << io->counter - io_start << endl;

        print_noise_budget_vec(enc_result);
        // print_ct(enc_result[0], data.slot_count);

        auto HE_result = search_postprocess(enc_result, data);

        // HACK
        // for (int i = 0; i < 128; i++) {
        //     cout << ((int64_t) HE_result[i] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
        // }

        // save HE_result to file
        std::string fileName = "scores.txt";

        // Open the file in write mode
        std::ofstream outFile(fileName);

        // Check if the file is open
        if (outFile.is_open()) {
            for (size_t i = 0; i < output_dim; ++i) {
                outFile << HE_result[i];
                if (i < output_dim - 1) {
                    outFile << ","; // Add a comma except after the last element
                }
            }
            outFile.close();
            std::cout << "[Client] Vector saved to " << fileName << std::endl;
        } else {
            std::cerr << "Error: Unable to open file " << fileName << std::endl;
        }
        
        delete[] HE_result;
    } else {
        // Server
        #ifdef HE_TIMING
        auto t1_total = high_resolution_clock::now();
        #endif

        auto io_start = io->counter;
        // vector<Ciphertext> cts(data.image_size * data.filter_h / data.slot_count);
        vector<Ciphertext> cts(1);
        recv_encrypted_vector(this->context, io, cts);

        cout << "[Server] Input cts received" << endl;

        // vector<uint64_t> vec(common_dim);
        // for (int i = 0; i < common_dim; i++) {
        //     vec[i] = B[i][0];
        // }

        #ifdef HE_TIMING
        auto t1_preprocess = high_resolution_clock::now();
        #endif

        cout << "[Server] Preprocessing Start" << endl;

        vector<uint64_t *> matrix_mod_p1(common_dim * output_dim);

        for (int i = 0; i < output_dim; i++) {
            matrix_mod_p1[i] = new uint64_t[common_dim];
            for (int j = 0; j < common_dim; j++) {
                matrix_mod_p1[i][j] = neg_mod((int64_t)B1[i][j], (int64_t)prime_mod);
            }
        }

        cout << "[Server] Preprocessing Packing Start" << endl;

        vector<Plaintext> dbembedding_packing = search_packing_db(matrix_mod_p1.data(), data);

        // print_pt(dbembedding_packing[dbembedding_packing.size() - 1], data.slot_count);

        vector<Plaintext> mask_packing = search_packing_mask(data);

        // cout << "[Server] Noise processed" << endl;
        #ifdef HE_TIMING
        auto t2_preprocess = high_resolution_clock::now();
        auto interval = (t2_preprocess - t1_preprocess)/1e+9;
        cout << "[Server] Preprocessing takes " << interval.count() << "sec" << endl;
        #endif

        #ifdef HE_DEBUG
            print_noise_budget_vec(cts);
        #endif

        // cout << "[Server] debugging " << dbembedding_packing.size() << " " << mask_packing.size() << endl;


        vector<Ciphertext> Cipher_plain_results(data.filter_w / data.slot_count);
        #ifdef HE_TIMING
        auto t1_cipher_plain = high_resolution_clock::now();
        #endif 

        // cout << "[Server] debugging result size " << Cipher_plain_results.size() << endl;

        search_inner_prod(cts, dbembedding_packing, mask_packing, data, Cipher_plain_results);

        #ifdef HE_TIMING
        auto t2_cipher_plain = high_resolution_clock::now();
        interval = (t2_cipher_plain - t1_cipher_plain)/1e+9;
        cout << "[Server] Cipher-Plaintext Inner Prod takes " << interval.count() << "sec" << endl;
        #endif

        send_encrypted_vector(io, Cipher_plain_results);

        cout << "[Server] Result sent" << endl;
        cout << "[Server] size of result (Bytes): " << io->counter - io_start << endl;

        for (int i = 0; i < output_dim; i++) {
            delete[] matrix_mod_p1[i];
        }

        #ifdef HE_TIMING
        auto t2_total = high_resolution_clock::now();
        interval = (t2_total - t1_total)/1e+9;
        cout << "[Server] Total Time " << interval.count() << "sec" << endl;
        #endif 
    }
}