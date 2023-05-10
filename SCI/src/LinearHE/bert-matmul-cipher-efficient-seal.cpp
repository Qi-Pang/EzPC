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

#include "LinearHE/bert-matmul-cipher-efficient-seal.h"
#include <omp.h>

using namespace std;
using namespace sci;
using namespace seal;

#define HE_TIMING

void BEFCField::print_noise_budget_vec(vector<Ciphertext> v) {
    cout << "Noise budget: ";
    for(int i = 0; i < v.size(); i++){
        cout << YELLOW << decryptor->invariant_noise_budget(v[i]) << " ";
    }
    cout << RESET << endl;
}

void BEFCField::print_ct(Ciphertext &ct, int len){
    Plaintext pt;
    decryptor->decrypt(ct, pt);
    print_pt(pt, len);
}

void BEFCField::print_pt(Plaintext &pt, int len) {
    vector<int64_t> dest(len, 0ULL);
    encoder->decode(pt, dest);
    cout << "Decode first 5 result: ";
    int non_zero_count;
    for(int i =0; i < 5; i++){
        cout << dest[i] << " ";
        // if(dest[i] != 0){
        //     non_zero_count += 1;
        // }
    }
    // cout << "Non zero count: " << non_zero_count;
    cout << endl;
}

// Generate the masks for 1-step rotation
vector<vector<Plaintext>> BEFCField::generate_rotation_masks(const FCMetadata &data) {
    vector<vector<Plaintext>> result;
    for (int i = 0; i < 128; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        vector<int64_t> mask2(data.slot_count, 0LL);
        for (int j = 0; j < 128 - i; j++) {
            for (int k = 0; k < 32; k++) {
                mask1[j + 128 * k] = 1;
                mask1[j + 128 * k + data.slot_count / 2] = 1;
            }
        }
        for (int j = 128 - i; j < 128; j++) {
            for (int k = 0; k < 32; k++) {
                mask2[j + 128 * k] = 1;
                mask2[j + 128 * k + data.slot_count / 2] = 1;
            }
        }
        Plaintext pt1;
        Plaintext pt2;
        encoder->encode(mask1, pt1);
        encoder->encode(mask2, pt2);
        result.push_back({pt1, pt2});
    }
    return result;
}

// Generate cipher_masks: 1111100000..., 0000011111..., ...
vector<Plaintext> BEFCField::generate_cipher_masks(const FCMetadata &data) {
    vector<Plaintext> result;
    for (int i = 0; i < 32; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128; k++)
            mask1[i * 128 + k] = 1;
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }

    for (int i = 0; i < 32; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128; k++)
            mask1[i * 128 + k + data.slot_count / 2] = 1;
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }

    for (int i = 0; i < 32; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128; k++)
            mask1[i * 128 + k] = 1;
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }

    for (int i = 0; i < 32; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128; k++)
            mask1[i * 128 + k + data.slot_count / 2] = 1;
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }
    return result;
}

// Generate packing_masks: 1111100000, 0000011111
vector<Plaintext> BEFCField::generate_packing_masks(const FCMetadata &data) {
    vector<Plaintext> result;
    vector<int64_t> mask1(data.slot_count, 0LL);
    vector<int64_t> mask2(data.slot_count, 0LL);
    for (int i = 0; i < data.slot_count / 2; i++) {
        mask1[i] = 1;
        mask2[i + data.slot_count / 2] = 1;
        
    }
    Plaintext pt1;
    Plaintext pt2;
    encoder->encode(mask1, pt1);
    encoder->encode(mask2, pt2);
    result.push_back(pt1);
    result.push_back(pt2);
    return result;
}

vector<Plaintext> BEFCField::generate_depth3_masks(const FCMetadata &data) {
    vector<Plaintext> result;

    for (int i = 0; i < 64; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128 - i; k++)
            mask1[i * 128 + k] = 1;
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }

    for (int i = 0; i < 64; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128 - i; k++)
            mask1[i * 128 + k] = 1;
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }

    for (int i = 0; i < 64; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 128 - i; k < 128; k++)
            mask1[i * 128 + k] = 1;
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }

    for (int i = 0; i < 64; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 128 - i; k < 128; k++)
            mask1[i * 128 + k] = 1;
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }
    return result;
}

Ciphertext BEFCField::rotation_by_one(const FCMetadata &data, Ciphertext ct, int k, vector<vector<Plaintext>> rotation_masks) {

    int m = -(128 - k);
    Ciphertext ct1;
    Ciphertext ct2;
    evaluator->rotate_rows(ct, k, *gal_keys, ct1);
    evaluator->rotate_rows(ct, m, *gal_keys, ct2);

    Ciphertext mul1;
    Ciphertext mul2;
    evaluator->multiply_plain(ct1, rotation_masks[k][0], mul1);
    evaluator->multiply_plain(ct2, rotation_masks[k][1], mul2);

    Ciphertext add;
    evaluator->add(mul1, mul2, add);

    return add;
}

vector<Ciphertext> BEFCField::rotation_by_one_depth3(const FCMetadata &data, Ciphertext ct, int k) {

    int m = -(128 - k);
    Ciphertext ct1;
    Ciphertext ct2;
    evaluator->rotate_rows(ct, k, *gal_keys, ct1);
    evaluator->rotate_rows(ct, m, *gal_keys, ct2);

    return {ct1, ct2};
}

// column-wise packing
vector<Ciphertext> BEFCField::bert_efficient_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data) {

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

vector<vector<Plaintext>> BEFCField::bert_efficient_preprocess_matrix(const uint64_t *const *matrix, const FCMetadata &data) {

    vector<vector<Plaintext>> weightMatrix;
    vector<int64_t> temp2;
    int num_diag = 32;
    int num_matrix_per_diag = data.filter_h / (data.slot_count / data.image_size); // should be 12
    for (int l = 0; l < num_diag; l++) {
        //iterate over all diagonals
        vector<Plaintext> temp_matrix_diag(num_matrix_per_diag);
        int matrix_diag_index = 0;
        for (int i = 0; i < data.filter_h / num_diag; i++) {
            //iterate over subblocks (32x32) of rows
            for (int j = 0; j < num_diag; j++) {
                //iterate over columns
                for (int k = 0; k < 128; k++) {
                    temp2.push_back(matrix[i * num_diag + j][(j + l) % num_diag]);
                }
                if (temp2.size() % (data.slot_count / 2) == 0) {
                    rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - l * 128, temp2.end());
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
        weightMatrix.push_back(temp_matrix_diag);
    }

    for (int l = 0; l < num_diag; l++) {
        //iterate over all diagonals
        vector<Plaintext> temp_matrix_diag(num_matrix_per_diag);
        int matrix_diag_index = 0;
        for (int i = 0; i < data.filter_h / num_diag; i++) {
            //iterate over subblocks (32x32) of rows
            for (int j = 0; j < num_diag; j++) {
                //iterate over columns
                for (int k = 0; k < 128; k++) {
                    temp2.push_back(matrix[i * num_diag + j][(j + l) % num_diag + num_diag]);
                }
                if (temp2.size() % (data.slot_count / 2) == 0) {
                    std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - l * 128, temp2.end());
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
        weightMatrix.push_back(temp_matrix_diag);
    }
    return weightMatrix;
}

/* Generates a masking vector of random noise that will be applied to parts of
 * the ciphertext that contain leakage */
Ciphertext BEFCField::bert_efficient_preprocess_noise(const uint64_t *secret_share, const FCMetadata &data) {
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

vector<Ciphertext> BEFCField::bert_efficient_online(vector<Ciphertext> &cts, vector<vector<Plaintext>> &enc_mat1, vector<vector<Plaintext>> &enc_mat2, const FCMetadata &data, vector<vector<Plaintext>> & rotation_masks) {

    //prepare rotated intermediate representation
    cout << "[Server] Online Start" << endl;
    vector<vector<Ciphertext>> rotatedIR(cts.size());
    int num_diag = 32;

    #ifdef HE_TIMING
    auto t1_rotate = high_resolution_clock::now();
    # endif

    #pragma omp parallel for
    for (int i = 0; i < cts.size(); i++) {   
        vector<Ciphertext> tmp;
        tmp.push_back(cts[i]);

        for (int j = 1; j < num_diag; j++) {
            Ciphertext temp_rot;
            evaluator->rotate_rows(cts[i], (num_diag - j) * data.image_size, *gal_keys, temp_rot);
            tmp.push_back(temp_rot);
        }
        rotatedIR[i] = tmp;
        tmp.clear();
    }

    #ifdef HE_TIMING
    auto t2_rotate = high_resolution_clock::now();
    auto interval = (t2_rotate - t1_rotate)/1e+9;
    std::cout << "[Server] Online - rotation takes " << interval.count() << "sec" << endl;
    # endif

    #ifdef HE_DEBUG
    print_noise_budget_vec(rotatedIR[0]);
    #endif

    #ifdef HE_TIMING
    auto t1_mul = high_resolution_clock::now();
    # endif

    //compute matrix multiplication
    vector<Ciphertext> temp_result1(enc_mat1.size() * cts.size());
    vector<Ciphertext> temp_result2(enc_mat2.size() * cts.size());

    #pragma omp parallel for
    for (int k = 0; k < enc_mat1.size() / 2 * cts.size(); k++) {
        //iterate over all ciphertexts
        int j = k / cts.size();
        int i = k % cts.size();
        Ciphertext ct1_l;
        Ciphertext ct1_r;
        evaluator->multiply_plain(rotatedIR[i][j], enc_mat1[j][i], ct1_l);
        evaluator->multiply_plain(rotatedIR[i][j], enc_mat1[j + enc_mat1.size() / 2][i], ct1_r);
        temp_result1[k] = ct1_l; // left half
        temp_result1[k + enc_mat1.size() * cts.size() / 2] = ct1_r; // right half

        Ciphertext ct2_l;
        Ciphertext ct2_r;
        evaluator->multiply_plain(rotatedIR[i][j], enc_mat2[j][i], ct2_l);
        evaluator->multiply_plain(rotatedIR[i][j], enc_mat2[j + enc_mat2.size() / 2][i], ct2_r);
        temp_result2[k] = ct2_l; // left half
        temp_result2[k + enc_mat2.size() * cts.size() / 2] = ct2_r; // right half
    }

    #ifdef HE_TIMING
    auto t2_mul = high_resolution_clock::now();
    interval = (t2_rotate - t1_rotate)/1e+9;
    std::cout << "[Server] Mult first layer takes " << interval.count() << "sec" << endl;
    # endif

    #ifdef HE_DEBUG
        print_noise_budget_vec(temp_result1);
        print_noise_budget_vec(temp_result2);
    #endif

    #ifdef HE_TIMING
    auto t1_add_rotate = high_resolution_clock::now();
    # endif

    Ciphertext result_left1 = temp_result1[0];
    Ciphertext result_left2 = temp_result2[0];

    for (int k = 1; k < temp_result1.size() / 2; k++) {
        evaluator->add_inplace(result_left1, temp_result1[k]);
        evaluator->add_inplace(result_left2, temp_result2[k]);
    }
    Ciphertext rtt_left1;
    Ciphertext rtt_left2;
    evaluator->rotate_columns(result_left1, *gal_keys, rtt_left1);
    evaluator->rotate_columns(result_left2, *gal_keys, rtt_left2);
    evaluator->add_inplace(result_left1, rtt_left1);
    evaluator->add_inplace(result_left2, rtt_left2);

    Ciphertext result_right1 = temp_result1[temp_result1.size() / 2];
    Ciphertext result_right2 = temp_result2[temp_result1.size() / 2];

    for (int k = 1; k < temp_result1.size() / 2; k++) {
        evaluator->add_inplace(result_right1, temp_result1[k + temp_result1.size() / 2]);
        evaluator->add_inplace(result_right2, temp_result2[k + temp_result2.size() / 2]);
    }

    Ciphertext rtt_right1;
    Ciphertext rtt_right2;
    evaluator->rotate_columns(result_right1, *gal_keys, rtt_right1);
    evaluator->rotate_columns(result_right2, *gal_keys, rtt_right2);
    evaluator->add_inplace(result_right1, rtt_right1);
    evaluator->add_inplace(result_right2, rtt_right2);

    #ifdef HE_TIMING
    auto t2_add_rotate = high_resolution_clock::now();
    interval = (t2_add_rotate - t1_add_rotate)/1e+9;
    std::cout << "[Server] Rotate and add takes " << interval.count() << "sec" << endl;
    # endif

    #ifdef HE_DEBUG
        PRINT_NOISE_BUDGET(decryptor, result_left1, "2nd Rotation and Add");
        PRINT_NOISE_BUDGET(decryptor, result_left2, "2nd Rotation and Add");
        PRINT_NOISE_BUDGET(decryptor, result_right1, "2nd Rotation and Add");
        PRINT_NOISE_BUDGET(decryptor, result_right2, "2nd Rotation and Add");
    #endif

    vector<Ciphertext> result = {result_left1, result_right1, result_left2, result_right2};

    return result;
}

// 1. rotate rhs for 128 x 1-step rotations
// 2. mult with lhs (producing 128 cts)
// 3. for each of the 128 cts, rotate for log(32) times, sum together + 1 time batch rotation
// 4. mult masks (1, 0 (x31), 1, 0 (x31), ... ) and sum together (do the first 32 (1st batch) and then the second batch).

vector<Ciphertext> BEFCField::bert_efficient_cipher(const FCMetadata &data, vector<Ciphertext> Cipher_plain_result, vector<vector<Plaintext>>& rotation_masks, vector<Plaintext>& cipher_masks)
{
    cout << "Entering bert_efficient_cipher" << endl;
    auto HE_result1_left = Cipher_plain_result[0];
    auto HE_result1_right = Cipher_plain_result[1];
    auto HE_result2_left = Cipher_plain_result[2];
    auto HE_result2_right = Cipher_plain_result[3];
    vector<Ciphertext> rotation_results(data.image_size);
    vector<Ciphertext> rotation_results_left(data.image_size);
    vector<Ciphertext> rotation_results_right(data.image_size);
    
    #ifdef HE_TIMING
    auto t1_mul = high_resolution_clock::now();
    # endif

    int num_diag = 32;
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++) {

        Ciphertext temp_mult = rotation_by_one(data, HE_result2_left, i, rotation_masks);
        Ciphertext ct_l;

        evaluator->multiply(HE_result1_left, temp_mult, ct_l);
        evaluator->relinearize_inplace(ct_l, *relin_keys);
        rotation_results_left[i] = ct_l;
        temp_mult = rotation_by_one(data, HE_result2_right, i, rotation_masks);
        Ciphertext ct_r;
        evaluator->multiply(HE_result1_right, temp_mult, ct_r);
        evaluator->relinearize_inplace(ct_r, *relin_keys);
        rotation_results_right[i] = ct_r;
        Ciphertext add;
        evaluator->add(ct_l, ct_r, add);
        rotation_results[i] = add;
    }

    #ifdef HE_TIMING
    auto t2_mul = high_resolution_clock::now();
    auto interval = (t2_mul - t1_mul)/1e+9;
    std::cout << "[Server] Cipher-Cipher Rotation 1 takes " << interval.count() << "sec" << endl;
    # endif

    #ifdef HE_DEBUG
    print_noise_budget_vec(rotation_results_left);
    print_noise_budget_vec(rotation_results_right);
    cout << "After cipher cipher mult before Add " << endl;
    print_noise_budget_vec(rotation_results);
    #endif

    #ifdef HE_TIMING
    auto t1_mul_2 = high_resolution_clock::now();
    # endif

    int local_rotation = std::ceil(std::log2(32));
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++) {
        for (int k = 0; k < local_rotation; k++) {
            Ciphertext temp2;
            evaluator->rotate_rows(rotation_results[i], (int32_t) pow(2, k) * 128, *gal_keys,temp2);
            evaluator->add_inplace(rotation_results[i], temp2);
        }

        evaluator->multiply_plain_inplace(rotation_results[i], cipher_masks[i]);

    }

    #ifdef HE_TIMING
    auto t2_mul_2 = high_resolution_clock::now();
    interval = (t2_mul_2 - t1_mul_2)/1e+9;
    std::cout << "[Server] Cipher-Cipher Rotation 2 takes " << interval.count() << "sec" << endl;
    # endif

    #ifdef HE_DEBUG
    cout << "After last mult" << endl;
    print_noise_budget_vec(rotation_results);
    #endif

    // Packing
    vector<Ciphertext> results(2);
    results[0] = rotation_results[0];
    results[1] = rotation_results[data.slot_count / data.image_size];

    #ifdef HE_TIMING
    auto t1_pack = high_resolution_clock::now();
    # endif

    #pragma omp parallel for
    for (int i = 1; i < data.slot_count / data.image_size; i++) {
        evaluator->add_inplace(results[0], rotation_results[i]);
        evaluator->add_inplace(results[1], rotation_results[i + data.slot_count / data.image_size]);
    }

    #ifdef HE_TIMING
    auto t2_pack = high_resolution_clock::now();
    interval = (t2_pack - t1_pack)/1e+9;
    std::cout << "[Server] Packing takes " << interval.count() << "sec" << endl;
    # endif

    #ifdef HE_DEBUG
    print_noise_budget_vec(results);
    #endif

    return results;
}

vector<Ciphertext> BEFCField::bert_efficient_cipher_depth3(const FCMetadata &data, vector<Ciphertext> Cipher_plain_result, vector<Plaintext>& depth3_masks){
    
    Ciphertext HE_result1_left = Cipher_plain_result[0];
    Ciphertext HE_result1_right = Cipher_plain_result[1];
    Ciphertext HE_result2_left = Cipher_plain_result[2];
    Ciphertext HE_result2_right = Cipher_plain_result[3];

    vector<Ciphertext> rotation_results(data.image_size * 2);
    auto t1 = high_resolution_clock::now();
    vector<Ciphertext> rotation_results_left(data.image_size * 2);
    vector<Ciphertext> rotation_results_right(data.image_size * 2);

    #ifdef HE_TIMING
    auto t1_mul = high_resolution_clock::now();
    # endif

    int num_diag = 32;
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++) {
        vector<Ciphertext> temp_mult = rotation_by_one_depth3(data, HE_result2_left, i);
        rotation_results_left[i] = HE_result1_left;
        evaluator->multiply_inplace(rotation_results_left[i], temp_mult[0]);
        evaluator->relinearize_inplace(rotation_results_left[i], *relin_keys);
        rotation_results_left[i + data.image_size] = HE_result1_left;
        evaluator->multiply_inplace(rotation_results_left[i + data.image_size], temp_mult[1]);
        evaluator->relinearize_inplace(rotation_results_left[i + data.image_size], *relin_keys);

        temp_mult = rotation_by_one_depth3(data, HE_result2_right, i);
        rotation_results_right[i] = HE_result1_right;
        evaluator->multiply_inplace(rotation_results_right[i], temp_mult[0]);
        evaluator->relinearize_inplace(rotation_results_right[i], *relin_keys);
        rotation_results_right[i + data.image_size] = HE_result1_right;
        evaluator->multiply_inplace(rotation_results_right[i + data.image_size], temp_mult[1]);
        evaluator->relinearize_inplace(rotation_results_right[i + data.image_size], *relin_keys);

        evaluator->add(rotation_results_left[i], rotation_results_right[i], rotation_results[i]);
        evaluator->add(rotation_results_left[i + data.image_size], rotation_results_right[i+ data.image_size], rotation_results[i+ data.image_size]);
    }

    #ifdef HE_TIMING
    auto t2_mul = high_resolution_clock::now();
    auto interval = (t2_mul - t1_mul)/1e+9;
    std::cout << "[Server] Cipher-Cipher Rotation 1 takes " << interval.count() << "sec" << endl;
    # endif

    #ifdef HE_TIMING
    auto t1_mul_2 = high_resolution_clock::now();
    # endif
    int local_rotation = std::ceil(std::log2(32));
    #pragma omp parallel for
    for (int i = 0; i < data.image_size * 2; i++) {
        for (int k = 0; k < local_rotation; k++) {
            Ciphertext temp2;
            evaluator->rotate_rows(rotation_results[i], (int32_t) pow(2, k) * 128, *gal_keys,temp2);
            evaluator->add_inplace(rotation_results[i], temp2);
        }
        evaluator->multiply_plain_inplace(rotation_results[i], depth3_masks[i]);
    }
    #ifdef HE_TIMING
    auto t2_mul_2 = high_resolution_clock::now();
    interval = (t2_mul_2 - t1_mul_2)/1e+9;
    std::cout << "[Server] Cipher-Cipher Rotation 2 takes " << interval.count() << "sec" << endl;
    # endif

    // Packing
    vector<Ciphertext> results(2);
    results[0] = rotation_results[0];
    results[1] = rotation_results[data.slot_count / data.image_size];

    #ifdef HE_TIMING
    auto t1_pack = high_resolution_clock::now();
    # endif

    #pragma omp parallel for
    for (int i = 1; i < data.slot_count / data.image_size; i++) {
        evaluator->add_inplace(results[0], rotation_results[i]);
        evaluator->add_inplace(results[0], rotation_results[i + data.image_size]);
        evaluator->add_inplace(results[1], rotation_results[i + data.slot_count / data.image_size]);
        evaluator->add_inplace(results[1], rotation_results[i + data.slot_count / data.image_size + data.image_size]);
    }

    #ifdef HE_TIMING
    auto t2_pack = high_resolution_clock::now();
    interval = (t2_pack - t1_pack)/1e+9;
    std::cout << "[Server] Packing takes " << interval.count() << "sec" << endl;
    # endif

    return results;
}

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

uint64_t* BEFCField::bert_efficient_postprocess(vector<Ciphertext> &cts, const FCMetadata &data) {
  uint64_t *result = new uint64_t[data.image_size*data.image_size];
  for (int i = 0; i < cts.size(); i++) {
    vector<int64_t> plain(data.slot_count, 0ULL);
    Plaintext pt;
    decryptor->decrypt(cts[i], pt);
    encoder->decode(pt, plain);

    #pragma omp parallel for
    for (int row = 0; row < data.slot_count; row++) {
        int j = row / data.image_size + i * 64;
        int k = row % data.image_size;
        // k-th row, j-th diag
        // k, (k + j) % 128
        result[k + ((k + j) % 128) * data.image_size] = plain[row];

    }
  }
  return result;
}

BEFCField::BEFCField(int party, NetIO *io) {
    this->party = party;
    this->io = io;
    this->slot_count = 8192;

    this->party = party;
    this->io = io;
    this->slot_count = POLY_MOD_DEGREE;
    generate_new_keys(party, io, slot_count, context, encryptor, decryptor,
                    evaluator, encoder, gal_keys, relin_keys, zero);
}

BEFCField::~BEFCField() {
    free_keys(party, encryptor, decryptor, evaluator, encoder, gal_keys, zero);
}

void BEFCField::configure() {
  data.slot_count = 8192;
  // Only works with a ciphertext that fits in a single ciphertext
  assert(data.slot_count >= data.image_size);

  data.filter_size = data.filter_h * data.filter_w;
  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / next_pow2(data.filter_w);
  // How many total ciphertexts we'll need
  data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
}

vector<uint64_t> BEFCField::ideal_functionality(uint64_t *vec, uint64_t **matrix) {
  vector<uint64_t> result(data.filter_h, 0ULL);
  for (int row = 0; row < data.filter_h; row++) {
    for (int idx = 0; idx < data.filter_w; idx++) {
      uint64_t partial = vec[idx] * matrix[row][idx];
      result[row] = result[row] + partial;
    }
  }
  return result;
}

void BEFCField::matrix_multiplication(int32_t input_dim, 
                                      int32_t common_dim, 
                                      int32_t output_dim, 
                                      vector<vector<uint64_t>> &A, 
                                      vector<vector<uint64_t>> &B, 
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

        auto cts = bert_efficient_preprocess_vec(vec, data);
        auto io_start = io->counter;
        send_encrypted_vector(io, cts);
        cout << "[Client] Input cts sent" << endl;
        cout << "[Client] Size of cts (Bytes): " << sizeof(Ciphertext) << " " << sizeof(Ciphertext) * cts.size() << endl;

        vector<Ciphertext> enc_result(2);
        recv_encrypted_vector(context, io, enc_result);
        cout << "[Client] Output cts received" << endl;
        cout << "[Client] size of cts (Bytes): " << io->counter - io_start << endl;

        print_ct(enc_result[0], data.slot_count);

        auto HE_result = bert_efficient_postprocess(enc_result, data);
        
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
        auto t1_preprocess = high_resolution_clock::now();
        #endif 

        vector<uint64_t> vec(common_dim);
        for (int i = 0; i < common_dim; i++) {
            vec[i] = B[i][0];
        }
        vector<uint64_t *> matrix_mod_p(common_dim);
        vector<uint64_t *> matrix(common_dim);
        for (int i = 0; i < common_dim; i++) {
        matrix_mod_p[i] = new uint64_t[output_dim];
        matrix[i] = new uint64_t[output_dim];
        for (int j = 0; j < output_dim; j++) {
            matrix_mod_p[i][j] = neg_mod((int64_t)B[i][j], (int64_t)prime_mod);
            int64_t val = (int64_t)B[i][j];
            if (val > int64_t(prime_mod/2)) {
            val = val - prime_mod;
            }
            matrix[i][j] = val;
        }
        }

        PRG128 prg;
        uint64_t *secret_share = new uint64_t[input_dim*output_dim];
        prg.random_mod_p<uint64_t>(secret_share, input_dim*output_dim, prime_mod);
        auto encoded_mat = bert_efficient_preprocess_matrix(matrix_mod_p.data(), data);

        auto rotation_masks = generate_rotation_masks(data);
        auto cipher_masks = generate_cipher_masks(data);
        auto depth3_masks = generate_depth3_masks(data);

        // Ciphertext enc_noise = bert_efficient_preprocess_noise(secret_share, data, cryptoContext_, keyPair_);
        // cout << "[Server] Noise processed" << endl;
        #ifdef HE_TIMING
        auto t2_preprocess = high_resolution_clock::now();
        auto interval = (t2_preprocess - t1_preprocess)/1e+9;
        cout << "[Server] Preprocessing takes " << interval.count() << "sec" << endl;
        #endif 

        auto io_start = io->counter;
        vector<Ciphertext> cts(12);
        recv_encrypted_vector(this->context, io, cts);

        #ifdef HE_DEBUG
            print_noise_budget_vec(cts);
        #endif

        #ifdef HE_TIMING
        auto t1_cipher_plain = high_resolution_clock::now();
        #endif 

        auto Cipher_plain_results = bert_efficient_online(cts, encoded_mat, encoded_mat, data, rotation_masks);

        #ifdef HE_TIMING
        auto t2_cipher_plain = high_resolution_clock::now();
        interval = (t2_cipher_plain - t1_cipher_plain)/1e+9;
        cout << "[Server] Cipher-Plaintext Matmul takes " << interval.count() << "sec" << endl;

        auto t1_cipher_cipher = high_resolution_clock::now();
        #endif 

        // auto HE_result = bert_efficient_cipher(data, Cipher_plain_results, rotation_masks, cipher_masks);
        auto HE_result = bert_efficient_cipher_depth3(data, Cipher_plain_results, depth3_masks);

        #ifdef HE_TIMING
        auto t2_cipher_cipher = high_resolution_clock::now();
        interval = (t2_cipher_cipher - t1_cipher_cipher)/1e+9;
        cout << "[Server] Cipher-Cipher Matmul takes " << interval.count() << "sec" << endl;
        #endif 

        send_encrypted_vector(io, HE_result);

        cout << "[Server] Result sent" << endl;
        cout << "[Server] size of result (Bytes): " << io->counter - io_start << endl;

        for (int i = 0; i < common_dim; i++) {
            delete[] matrix_mod_p[i];
            delete[] matrix[i];
        }
        delete[] secret_share;

        #ifdef HE_TIMING
        auto t2_total = high_resolution_clock::now();
        interval = (t2_total - t1_total)/1e+9;
        cout << "[Server] Total Time " << interval.count() << "sec" << endl;
        #endif 
    }
}