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
// #define HE_DEBUG

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
    vector<uint64_t> dest(len, 0ULL);
    encoder->decode(pt, dest);
    cout << "Decode first 5 rows: ";
    int non_zero_count;
    for(int i = 0; i < 128; i++){
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
        for (int k = 0; k < 128 - i - 64; k++)
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
        for (int k = 128 - i - 64; k < 128; k++)
            mask1[i * 128 + k] = 1;
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }
    return result;
}

vector<Plaintext> BEFCField::generate_cross_packing_masks(const FCMetadata &data) {
    vector<Plaintext> result;

    for (int i = 0; i < 32; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        if (i == 0) {
            for (int k = 0; k < 128 - i; k++)
                mask1[k] = 1;
        }
        else {
            for (int k = 0; k < 128 - i; k++)
                mask1[i * 128 + k] = 1;
            for (int k = 0; k < 128 - i; k++)
                mask1[i * 128 + k + data.slot_count / 2] = 1;
        }
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }

    for (int i = 32; i <= 64; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        if (i == 64) {
            for (int k = 0; k < 128 - i; k++)
                mask1[k + data.slot_count / 2] = 1;
        }
        else {
            for (int k = 0; k < 128 - i; k++)
                mask1[(i - 32) * 128 + k] = 1;
            for (int k = 0; k < 128 - i; k++)
                mask1[(i - 32) * 128 + k + data.slot_count / 2] = 1;
        }
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }

    for (int i = 0; i < 32; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        if (i == 0) {
            for (int k = 128 - i; k < 128; k++)
                mask1[k] = 1;
        }
        else {
            for (int k = 128 - i; k < 128; k++)
                mask1[i * 128 + k] = 1;
            for (int k = 128 - i; k < 128; k++)
                mask1[i * 128 + k + data.slot_count / 2] = 1;
        }
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }
    for (int i = 32; i <= 64; i++) {
        vector<int64_t> mask1(data.slot_count, 0LL);
        if (i == 64) {
            for (int k = 128 - i; k < 128; k++)
                mask1[k + data.slot_count / 2] = 1;
        }
        else {
            for (int k = 128 - i; k < 128; k++)
                mask1[(i - 32) * 128 + k] = 1;
            for (int k = 128 - i; k < 128; k++)
                mask1[(i - 32) * 128 + k + data.slot_count / 2] = 1;
        }
        Plaintext pt;
        encoder->encode(mask1, pt);
        result.push_back(pt);
    }
    return result;
}

vector<Ciphertext> BEFCField::rotation_by_one_depth3(const FCMetadata &data, const Ciphertext &ct, int k) {

    int m = -(128 - k);
    Ciphertext ct1;
    Ciphertext ct2;
    evaluator->rotate_rows(ct, k, *gal_keys, ct1);
    evaluator->rotate_rows(ct, m, *gal_keys, ct2);

    return {ct1, ct2};
}

// column-wise packing
vector<Ciphertext> BEFCField::bert_efficient_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data) {

    vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
    vector<Ciphertext> cts;
    for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
    {
        pod_matrix = vector<uint64_t>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
        Ciphertext ct;
        Plaintext pt;
        encoder->encode(pod_matrix, pt);
        encryptor->encrypt(pt, ct);
        cts.push_back(ct);
    }
    return cts;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> BEFCField::bert_cross_packing_matrix(const uint64_t *const *matrix1, const uint64_t *const *matrix2, const FCMetadata &data) {
    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    vector<uint64_t> temp2;
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = data.filter_h / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag; // should be 8

    int n1;
    int n2;
    if (data.slot_count == 4096) {
        n1 = 4;
        n2 = 4;
    }
    else {
        n1 = 8;
        n2 = 4;
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
                            // HACK: verify sparsity
                            // cout << "packing" << endl;
                            // for (int temp2_ind = 0; temp2_ind < data.slot_count / data.image_size; temp2_ind++) {
                            //     cout << temp2[temp2_ind * data.image_size] << " ";
                            // }
                            // cout << endl;
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

vector<Plaintext> BEFCField::bert_cross_packing_bias(const uint64_t *matrix1, const uint64_t *matrix2, const uint64_t *matrix3, const FCMetadata &data) {
    vector<Plaintext> cross_bias_packing(3 * data.image_size * data.filter_w / data.slot_count);
    int matrix1_pointer = 0;
    int matrix2_pointer = 0;
    for (int packing_num = 0; packing_num < 2 * data.image_size * data.filter_w / data.slot_count; packing_num++) {
        vector<uint64_t> temp(data.slot_count, 0ULL);
        int right_flag = 0;
        int row = 0;
        while (row < data.slot_count) {
            if (row >= data.slot_count / 2) {
                right_flag = 1;
            }
            if (right_flag == 0) {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = matrix1[matrix1_pointer];
                }
                matrix1_pointer++;
            }
            else {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = matrix2[matrix2_pointer];
                }
                matrix2_pointer++;
            }
            row += data.image_size;
        }
        Plaintext pt;
        encoder->encode(temp, pt);
        cross_bias_packing[packing_num] = pt;
        temp.clear();
    }
    int matrix3_pointer1 = 0;
    int matrix3_pointer2 = data.filter_w / 2;
    for (int packing_num = 2 * data.image_size * data.filter_w / data.slot_count; packing_num < 3 * data.image_size * data.filter_w / data.slot_count; packing_num++) {

        vector<uint64_t> temp(data.slot_count, 0ULL);
        int row = 0;
        while (row < data.slot_count) {
            if (row < data.slot_count / 2) {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = matrix3[matrix3_pointer1];
                }
                matrix3_pointer1++;
                if (matrix3_pointer1 % (data.filter_w / 2) == 0)
                    matrix3_pointer1 += data.filter_w / 2;
            }
            else {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = matrix3[matrix3_pointer2];
                }
                matrix3_pointer2++;
                if (matrix3_pointer2 % (data.filter_w / 2) == 0)
                    matrix3_pointer2 += data.filter_w / 2;
            }
            row += data.image_size;
        }
        Plaintext pt;
        encoder->encode(temp, pt);
        cross_bias_packing[packing_num] = pt;
        temp.clear();
    }
    return cross_bias_packing;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> BEFCField::bert_cross_packing_single_matrix(const uint64_t *const *matrix1, const uint64_t *const *matrix2, const FCMetadata &data, const bool &softmax_V_flag) {
    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    vector<uint64_t> temp2;

    int n1;
    int n2;
    if (data.slot_count == 4096) {
        n1 = 4;
        n2 = 4;
    }
    else {
        n1 = 8;
        n2 = 4;
    }

    int weight_hight = data.filter_h;
    if (softmax_V_flag) {
        weight_hight = data.image_size;
        n1 = 4;
        n2 = 8;
    }
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = weight_hight / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag / 2; // should be 8

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(weight_hight * data.image_size / data.slot_count);
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
                            // HACK: verify sparsity
                            // cout << "packing" << endl;
                            // for (int temp2_ind = 0; temp2_ind < data.slot_count / data.image_size; temp2_ind++) {
                            //     cout << temp2[temp2_ind * data.image_size] << " ";
                            // }
                            // cout << endl;
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
            vector<Plaintext> temp_matrix_diag(weight_hight * data.image_size / data.slot_count);
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


void BEFCField::bert_cipher_plain_bsgs(const vector<Ciphertext> &cts, const vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &cross_mats, const vector<vector<Plaintext>> &Bias, const vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &cross_mats_single, const FCMetadata &data, vector<Ciphertext> &result) {

    auto t1 = high_resolution_clock::now();
    vector<vector<Ciphertext>> rotatedIR(cts.size()); // cts.size() = 48
    int n1;
    int n2;
    if (data.slot_count == 4096) {
        n1 = 4;
        n2 = 4;
    }
    else {
        n1 = 8;
        n2 = 4;
    }
    int num_diag = data.slot_count / data.image_size / 2;
    // vector<Ciphertext> result(data.image_size * data.filter_w / data.slot_count * 3 * 12);
    cout << "[Server] Online Start" << endl;
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
    t1 = high_resolution_clock::now();
    omp_set_nested(1);
    // #pragma omp parallel 
    // #pragma omp single
    #pragma omp parallel for num_threads(4)
    for (int packing_index = 0; packing_index < 12; packing_index++) {
        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w / data.slot_count * 3, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results1(data.image_size * data.filter_w / data.slot_count * 3, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Plaintext>> enc_mat1 = cross_mats[packing_index].first;
        vector<vector<Plaintext>> enc_mat2 = cross_mats[packing_index].second;
        vector<vector<Plaintext>> enc_mat3 = cross_mats_single[packing_index].first;
        vector<vector<Plaintext>> enc_mat4 = cross_mats_single[packing_index].second;

        #pragma omp parallel for num_threads(8)
        // #pragma omp taskloop
        for (int k = 0; k < cts.size() * n2; k++) {
            int j = k / cts.size();
            int ct_i = k % cts.size();
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count * 2; l++) {
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

            int third_index = data.image_size * data.filter_w / data.slot_count * 2;
            for (int l = third_index; l < data.image_size * data.filter_w / data.slot_count * 3; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct1_l;
                    Ciphertext ct1_r;
                    evaluator->multiply_plain(rotatedIR[ct_i][i], enc_mat3[n1 * j + i + (l - third_index) * num_diag][ct_i], ct1_l);
                    evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_mat4[n1 * j + i + (l - third_index) * num_diag][ct_i], ct1_r);
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

        #pragma omp parallel for num_threads(8)
        // #pragma omp taskloop
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
                for (int l = 0; l < data.image_size * data.filter_w / data.slot_count * 3; l++) {
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

        #pragma omp parallel for num_threads(8)
        // #pragma omp taskloop
        for (int l = 0; l < data.image_size * data.filter_w / data.slot_count * 3; l++) {
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
            result[l + packing_index * data.image_size * data.filter_w / data.slot_count * 3] = ct;
            evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w / data.slot_count * 3], Bias[packing_index][l]);
        }
    }

    t2 = high_resolution_clock::now();
    ms_double = (t2 - t1)/1e+9;
    cout << "[Server] Online Done " << ms_double.count() << endl;

}

// 1. rotate rhs for 128 x 1-step rotations
// 2. mult with lhs (producing 128 cts)
// 3. for each of the 128 cts, rotate for log(32) times, sum together + 1 time batch rotation
// 4. mult masks (1, 0 (x31), 1, 0 (x31), ... ) and sum together (do the first 32 (1st batch) and then the second batch).

void BEFCField::bert_cipher_cipher_cross_packing(const FCMetadata &data, const vector<Ciphertext> &Cipher_plain_result, const vector<Plaintext> &cross_masks, vector<Ciphertext> &results) {
    int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;

    #pragma omp parallel for num_threads(4)
    for (int packing_index = 0; packing_index < 12; packing_index++) {
        Ciphertext HE_result_1_left = Cipher_plain_result[0 + packing_gap * packing_index];
        Ciphertext HE_result_2_left = Cipher_plain_result[1 + packing_gap * packing_index];

        Ciphertext HE_result_1_right;
        Ciphertext HE_result_2_right;

        evaluator->rotate_columns(HE_result_1_left, *gal_keys, HE_result_1_right);
        evaluator->rotate_columns(HE_result_2_left, *gal_keys, HE_result_2_right);

        vector<Ciphertext> rotation_results(data.image_size + 2);
        auto t1 = high_resolution_clock::now();
        vector<Ciphertext> rotation_results_left(data.image_size + 2);
        vector<Ciphertext> rotation_results_right(data.image_size + 2);

        #pragma omp parallel for num_threads(16)
        for (int i = 0; i <= data.image_size / 2; i++) {
            vector<Ciphertext> temp_mult = rotation_by_one_depth3(data, HE_result_1_right, i);

            evaluator->multiply(HE_result_1_left, temp_mult[0], rotation_results_left[i]);
            // evaluator->relinearize_inplace(rotation_results_left[i], *relin_keys);

            evaluator->multiply(HE_result_1_left, temp_mult[1], rotation_results_left[i + data.image_size / 2 + 1]);
            // evaluator->relinearize_inplace(rotation_results_left[i + data.image_size / 2 + 1], *relin_keys);

            temp_mult = rotation_by_one_depth3(data, HE_result_2_right, i);

            evaluator->multiply(HE_result_2_left, temp_mult[0], rotation_results_right[i]);
            // evaluator->relinearize_inplace(rotation_results_right[i], *relin_keys);

            evaluator->multiply(HE_result_2_left, temp_mult[1], rotation_results_right[i + data.image_size / 2 + 1]);
            // evaluator->relinearize_inplace(rotation_results_right[i + data.image_size / 2 + 1], *relin_keys);

            evaluator->add(rotation_results_left[i], rotation_results_right[i], rotation_results[i]);
            evaluator->relinearize_inplace(rotation_results[i], *relin_keys);
            evaluator->add(rotation_results_left[i + data.image_size / 2 + 1], rotation_results_right[i + data.image_size / 2 + 1], rotation_results[i + data.image_size / 2 + 1]);
            evaluator->relinearize_inplace(rotation_results[i + data.image_size / 2 + 1], *relin_keys);

        }
        auto t2 = high_resolution_clock::now();
        auto ms_double = (t2 - t1)/1e+9;
        // std::cout << "[Server] Cipher-Cipher Rotation 1 " << ms_double.count() << std::endl;

        t1 = high_resolution_clock::now();
        int local_rotation = std::ceil(std::log2(32));
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < data.image_size + 2; i++) {
            for (int k = 0; k < local_rotation; k++) {
                Ciphertext temp2;
                evaluator->rotate_rows(rotation_results[i], (int32_t) pow(2, k) * 128, *gal_keys, temp2);
                evaluator->add_inplace(rotation_results[i], temp2);
            }
            evaluator->multiply_plain_inplace(rotation_results[i], cross_masks[i]);
        }
        t2 = high_resolution_clock::now();
        ms_double = (t2 - t1)/1e+9;
        // std::cout << "[Server] Cipher-Cipher Rotation 2 " << ms_double.count() << std::endl;
        // Packing
        t1 = high_resolution_clock::now();
        
        evaluator->add(rotation_results[0], rotation_results[65], results[0 + 2 * packing_index]);
        evaluator->add(rotation_results[32], rotation_results[32 + 65], results[1 + 2 * packing_index]);

        for (int i = 1; i < 32; i++) {
            Ciphertext temp;
            evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[i]);
            evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[i + 65]);
            evaluator->add_inplace(results[1 + 2 * packing_index], rotation_results[i + 32]);
            evaluator->add_inplace(results[1 + 2 * packing_index], rotation_results[i + 32 + 65]);
        }

        evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[64]);
        evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[64 + 65]);
        t2 = high_resolution_clock::now();
        ms_double = (t2 - t1)/1e+9;
        // std::cout << "[Server] Cipher-Cipher Packing " << ms_double.count() << std::endl;
    }
}

uint64_t* BEFCField::bert_efficient_postprocess(vector<Ciphertext> &cts, const FCMetadata &data) {
    uint64_t *result = new uint64_t[data.image_size * data.filter_w * 3];
    int num_cts_first_2 = data.image_size * data.filter_w * 2 / data.slot_count;
    for (int i = 0; i < num_cts_first_2; i++) {
        vector<int64_t> plain(data.slot_count, 0ULL);
        Plaintext pt;
        decryptor->decrypt(cts[i], pt);
        encoder->decode(pt, plain);

        #pragma omp parallel for
        for (int row = 0; row < data.slot_count; row++) {
            int right_flag = 0;
            if (row >= data.slot_count / 2) {
                right_flag = 1;
            }
            int j = row / data.image_size - right_flag * data.slot_count / 2 / data.image_size + i * data.slot_count / 2 / data.image_size + right_flag * data.filter_w;
            int k = row % data.image_size;
            result[k + j * data.image_size] = plain[row];
        }
    }

    for (int i = num_cts_first_2; i < cts.size() / 12; i++) {
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

uint64_t* BEFCField::bert_cross_packing_postprocess(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing) {
    uint64_t *result = new uint64_t[data.image_size*data.image_size*12];
    int num_cts_per_mat = data.image_size * data.image_size / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat; i++) {
            vector<uint64_t> plain(data.slot_count, 0ULL);
            Plaintext tmp;
            decryptor->decrypt(cts[i + packing_num * num_cts_per_mat], tmp);
            encoder->decode(tmp, plain);
            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (j < 32) { // k, (k + j) % 128
                        result[k + ((k + j + i * 32) % data.image_size) * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                    else if (j == 32 && i == 0) { // (64 + k) % 128, k
                        result[((k + 64) % data.image_size) + k * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                    else { // (k - 32 + j) % 128, k
                        result[k * data.image_size + (k + j - 32 + i * 32) % data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                }
            }
            else {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (j < 32) { // k, (k + j) % 128
                        result[k * data.image_size + ((k + j + i * 32) % data.image_size) + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                    else if (j == 32 && i == 0) { // (64 + k) % 128, k
                        result[((k + 64) % data.image_size) * data.image_size + k + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                    else { // (k - 32 + j) % 128, k
                        result[k + ((k + j - 32 + i * 32) % data.image_size) * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                }
            }
        }
    }
    return result;
}

vector<vector<vector<uint64_t>>> BEFCField::softmax_mask(const FCMetadata &data) {
    vector<vector<vector<uint64_t>>> mask(2, vector<vector<uint64_t>>(data.image_size, vector<uint64_t>(data.image_size)));
    for (int i = 0; i < data.image_size; i++) {
        vector<uint64_t> mask1(data.image_size, 0ULL);
        vector<uint64_t> mask2(data.image_size, 0ULL);
        for (int j = 0; j < data.image_size - i; j++) {
            mask1[j] = 1;
        }
        for (int j = data.image_size - i; j < data.image_size; j++) {
            mask2[j] = 1;
        }
        mask[0][i] = mask1;
        mask[1][i] = mask2;
    }
    return mask;
}

// matrix is row-packed with 12 * 128 rows and 128 cols
vector<vector<vector<Plaintext>>> BEFCField::preprocess_softmax_s2(const uint64_t *matrix, const FCMetadata &data, vector<vector<vector<uint64_t>>> &mask) {

    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size;
    vector<vector<vector<Plaintext>>> s2_pack(12);

    #pragma omp parallel for
    for (int packing_ind = 0; packing_ind < 12; packing_ind++) {
        vector<uint64_t> temp2;
        vector<vector<Plaintext>> weightMatrix1(2, vector<Plaintext>(num_diag));
        
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            vector<uint64_t> r1(data.image_size);
            vector<uint64_t> r2(data.image_size);
            for (int j = 0; j < num_diag; j++) {
                temp2.push_back(matrix[((j + diag_ind) % num_diag + packing_ind * data.image_size) * data.image_size + j]);
            }
            // std::rotate(temp2.begin(), temp2.begin() + temp2.size() - diag_ind, temp2.end());
            std::transform(temp2.begin(), temp2.end(), mask[0][diag_ind].begin(), r1.begin(), std::multiplies<uint64_t>());
            std::transform(temp2.begin(), temp2.end(), mask[1][diag_ind].begin(), r2.begin(), std::multiplies<uint64_t>());
            for (int j = 0; j < std::log2(num_diag_per_ct); j++) {
                r1.reserve(r1.size() + distance(r1.begin(), r1.end()));
                r1.insert(r1.end(), r1.begin(), r1.end());
                r2.reserve(r2.size() + distance(r2.begin(), r2.end()));
                r2.insert(r2.end(), r2.begin(), r2.end());
            }

            Plaintext pt;
            encoder->encode(r1, pt);
            weightMatrix1[0][diag_ind] = pt;
            encoder->encode(r2, pt);
            weightMatrix1[1][diag_ind] = pt;
            temp2.clear();
            r1.clear();
            r2.clear();
        }
        s2_pack[packing_ind] = weightMatrix1;
    }
    return s2_pack;
}

// matrix is row-packed with 12 * 128 rows and 128 cols
vector<Ciphertext> BEFCField::preprocess_softmax_s1(const uint64_t *matrix, const FCMetadata &data) {
    int num_ct_per_matrix = data.image_size * data.image_size / data.slot_count;
    vector<Ciphertext> enc_softmax(12 * num_ct_per_matrix);
    #pragma omp parallel for
    for (int packing_ind = 0; packing_ind < 12; packing_ind++) {
        for (int k = 0; k < data.image_size * data.image_size / data.slot_count; k++) {
            vector<uint64_t> pod_matrix(data.slot_count);
            for (int j = 0; j < data.slot_count / data.image_size; j++) {
                for (int i = 0; i < data.image_size; i++) {
                    pod_matrix[i + j * data.image_size] = matrix[(i + packing_ind * data.image_size) * data.image_size + j + data.slot_count / data.image_size * k];
                }
            }
            Ciphertext ct;
            Plaintext pt;
            encoder->encode(pod_matrix, pt);
            encryptor->encrypt(pt, ct);
            enc_softmax[packing_ind * num_ct_per_matrix + k] = ct;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < enc_softmax.size(); i++) {
        evaluator->mod_switch_to_next_inplace(enc_softmax[i]);
    }
    return enc_softmax;
}

// matrix is row-packed with 12 * 128 rows and 64 cols
vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> BEFCField::preprocess_softmax_v_r(const uint64_t *matrix, const FCMetadata &data) {
    vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> R_pack(12);

    #pragma omp parallel for
    for (int packing_ind = 0; packing_ind < 12; packing_ind++) {
        vector<uint64_t *> matrix_1(data.image_size);
        vector<uint64_t *> matrix_2(data.image_size);
        for (int i = 0; i < data.image_size; i++) {
            matrix_1[i] = new uint64_t[data.filter_w / 2];
            matrix_2[i] = new uint64_t[data.filter_w / 2];
            for (int j = 0; j < data.filter_w / 2; j++) {
                matrix_1[i][j] = matrix[(i + packing_ind * data.image_size) * data.filter_w + j];
                matrix_2[i][j] = matrix[(i + packing_ind * data.image_size) * data.filter_w + j + data.filter_w / 2];
            }
        }
        auto temp_R_pack = bert_cross_packing_single_matrix(matrix_1.data(), matrix_2.data(), data, true);
        R_pack[packing_ind] = temp_R_pack;
    }
    return R_pack;
}

uint64_t* BEFCField::client_S1_V_R(const uint64_t *softmax_s1, vector<Ciphertext> &V, const FCMetadata &data) {
    uint64_t* result = new uint64_t[12 * data.image_size * data.filter_w];
    // vector<vector<vector<uint64_t>>> result(12, vector<vector<uint64_t>> (data.image_size, vector<uint64_t> (data.filter_w)));
    auto V_R = bert_postprocess_V(V, data, true);
    int total_packing = 12;
    for (int packing_num = 0; packing_num < total_packing; packing_num++) {
        #pragma omp parallel for
        for(int i = 0; i < data.image_size; i++) {
            for(int j = 0; j < data.filter_w; j++) {
                result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = 0;
                for(int k = 0; k < data.image_size; k++) {
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] += neg_mod((int64_t)softmax_s1[(packing_num * data.image_size + i) * data.image_size + k] * V_R[k + j * data.image_size + data.image_size * data.filter_w * packing_num], (int64_t)prime_mod);
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = neg_mod((int64_t)result[packing_num * data.image_size * data.filter_w + i + j * data.image_size], (int64_t)prime_mod);
                }
            }
        }
    }
    return result;
}

void BEFCField::bert_softmax_V(vector<Ciphertext> &softmax_s1, vector<vector<vector<Plaintext>>> &softmax_s2, vector<Ciphertext> &V, vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &R, const FCMetadata &data, vector<Ciphertext> &result) {
    // FIXME: pack R according to ours ctxpt
    // FIXME: compute softmax_s1 x R

    #pragma omp parallel for
    for (int i = 0; i < V.size(); i++) {
        evaluator->mod_switch_to_next_inplace(V[i]);
    }

    int num_ct_per_s1 = data.image_size * data.image_size / data.slot_count;
    int num_ct_per_V = data.image_size * data.filter_w / data.slot_count;

    omp_set_nested(1);
    #pragma omp parallel for num_threads(4)
    for (int packing_ind = 0; packing_ind < 12; packing_ind++) {

        vector<vector<Plaintext>> R1 = R[packing_ind].first;
        vector<vector<Plaintext>> R2 = R[packing_ind].second;
        vector<vector<Ciphertext>> rotatedIR(num_ct_per_s1);
        int n1 = 4;
        int n2 = 8;
        int num_diag = data.slot_count / data.image_size / 2;
        auto t1 = high_resolution_clock::now();

        #pragma omp parallel for
        for (int i = 0; i < num_ct_per_s1; i++) {
            vector<Ciphertext> tmp;
            tmp.push_back(softmax_s1[packing_ind * num_ct_per_s1 + i]);

            for (int j = 1; j < n1; j++) {
                Ciphertext temp_rot;
                evaluator->rotate_rows(softmax_s1[packing_ind * num_ct_per_s1 + i], (num_diag - j) * data.image_size, *gal_keys, temp_rot);
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
        auto t2 = high_resolution_clock::now();
        auto ms_double = (t2 - t1)/1e+9;
        cout << "[Server] Online - rotation done " << ms_double.count() << endl;
        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results1(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2 * num_ct_per_s1));
        t1 = high_resolution_clock::now();

        #pragma omp parallel for
        for (int k = 0; k < num_ct_per_s1 * n2; k++) {
            int j = k / num_ct_per_s1;
            int ct_i = k % num_ct_per_s1;
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct1_l;
                    Ciphertext ct1_r;
                    evaluator->multiply_plain(rotatedIR[ct_i][i], R1[n1 * j + i + l * num_diag][ct_i], ct1_l);
                    evaluator->multiply_plain(rotatedIR[ct_i][i + n1], R2[n1 * j + i + l * num_diag][ct_i], ct1_r);
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
            for (int ct_i = 0; ct_i < num_ct_per_s1; ct_i++) {
                for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                    if (ct_i == 0)
                        temp_results[l][j] = temp_results1[l][j * num_ct_per_s1 + ct_i];
                    else
                        evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * num_ct_per_s1 + ct_i]);
                }
            }
        }

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
            result[packing_ind * data.image_size * data.filter_w / data.slot_count + l] = ct;
        }

        t2 = high_resolution_clock::now();
        ms_double = (t2 - t1)/1e+9;
        cout << "[Server] Online Done " << ms_double.count() << endl;

        // FIXME: pack softmax_s2 according to gazelle
        // FIXME: compute softmax_s2 x V

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_ct_per_V; ct_ind++) {
            vector<Ciphertext> rotation_results(num_diag);

            #pragma omp parallel for
            for (int i = 0; i < num_diag; i++) {
                Ciphertext temp1;
                Ciphertext temp2;
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(data, V[packing_ind * num_ct_per_V + ct_ind], i);
                evaluator->multiply_plain(temp_mult[0], softmax_s2[packing_ind][0][i], temp1);
                evaluator->multiply_plain(temp_mult[1], softmax_s2[packing_ind][1][i], temp2);
                evaluator->add(temp1, temp2, rotation_results[i]);
            }
            for (int i = 0; i < num_diag; i++) {
                evaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w / data.slot_count + ct_ind], rotation_results[i]);
            }
            rotation_results.clear();
        }
    }
    for (int i = 0; i < result.size(); i++) {
        evaluator->mod_switch_to_next_inplace(result[i]);
    }
}

uint64_t* BEFCField::bert_postprocess_V(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing) {
    // uint64_t *result_V = new uint64_t[data.image_size*data.filter_w*12];
    uint64_t *result_V = new uint64_t[cts.size() * data.slot_count];
    int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    for (int packing_num = 0; packing_num < total_packing; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            vector<uint64_t> plain(data.slot_count, 0ULL);
            Plaintext pt;
            decryptor->decrypt(cts[i + packing_num * num_cts_per_mat_V], pt);
            encoder->decode(pt, plain);
            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    result_V[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
            else {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    j += i * data.slot_count / data.image_size / 2;
                    result_V[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
    return result_V;
}

BEFCField::BEFCField(int party, NetIO *io) {
    this->party = party;
    this->io = io;
    this->slot_count = 8192;

    this->party = party;
    this->io = io;
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

void BEFCField::matrix_multiplication(int32_t input_dim, 
                                      int32_t common_dim, 
                                      int32_t output_dim, 
                                      vector<vector<uint64_t>> &A, 
                                      vector<vector<vector<uint64_t>>> &B1, 
                                      vector<vector<vector<uint64_t>>> &B2, 
                                      vector<vector<vector<uint64_t>>> &B3, 
                                      vector<vector<uint64_t>> &Bias1, 
                                      vector<vector<uint64_t>> &Bias2, 
                                      vector<vector<uint64_t>> &Bias3, 
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
                vec[j*input_dim + i] = neg_mod((int64_t)A[i][j], (int64_t)prime_mod);

        auto cts = bert_efficient_preprocess_vec(vec, data);
        auto io_start = io->counter;
        print_noise_budget_vec(cts);
        send_encrypted_vector(io, cts);
        cout << "[Client] Input cts sent" << endl;
        cout << "[Client] Size of cts (Bytes): " << sizeof(Ciphertext) << " " << sizeof(Ciphertext) * cts.size() << endl;

        vector<Ciphertext> enc_result(3 * 12);
        recv_encrypted_vector(context, io, enc_result);
        cout << "[Client] Output cts received" << endl;
        cout << "[Client] size of cts (Bytes): " << io->counter - io_start << endl;

        print_noise_budget_vec(enc_result);
        // print_ct(enc_result[0], data.slot_count);

        vector<Ciphertext> V = {enc_result.begin() + data.image_size * data.image_size / data.slot_count * 12, enc_result.end()};

        // auto HE_result = bert_efficient_postprocess(enc_result, data);
        auto HE_result = bert_cross_packing_postprocess(enc_result, data, true);
        // auto V_result = bert_postprocess_V(V, data, true);

        // HACK: verify
        // cout << "col packing" << endl;
        // for (int i = 0; i < 1; i++) {
        //     for (int j = 0; j < 64; j++)
        //         cout << ((int64_t) V_result[i + j * 128] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
        //     cout << endl;
        // }

        // HACK:
        // Below is computing softmax * V

        auto io_checkpoint = io->counter;

        uint64_t *softmax_s1 = new uint64_t[128 * 12 * 128];
        for (int i = 0; i < 128 * 12; i++) {
            for (int j = 0; j < 128; j++) {
                softmax_s1[i * 128 + j] = (i * 100 + j) % 1000;
            }
        }
        vector<Ciphertext> S1_pack = preprocess_softmax_s1(softmax_s1, data);

        auto S1_V_R = client_S1_V_R(softmax_s1, V, data);

        send_encrypted_vector(io, S1_pack);

        vector<Ciphertext> enc_softmax_V(12 * data.image_size * data.filter_w / data.slot_count);
        recv_encrypted_vector(context, io, enc_softmax_V);

        print_noise_budget_vec(enc_softmax_V);

        auto softmax_V = bert_postprocess_V(enc_softmax_V, data, true);

        for (int i = 0; i < data.image_size * data.filter_w * 12; i++) {
            softmax_V[i] += S1_V_R[i];
        }

        cout << "[Client] Result sent" << endl;
        cout << "[Client] size of result (Bytes): " << io->counter - io_checkpoint << endl;

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 64; j++)
                cout << ((int64_t) softmax_V[i + j * 128] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
            cout << endl;
        }

        #ifdef HE_DEBUG
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 128; j++)
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

        // vector<uint64_t> vec(common_dim);
        // for (int i = 0; i < common_dim; i++) {
        //     vec[i] = B[i][0];
        // }

        auto io_start = io->counter;
        vector<Ciphertext> cts(12);
        recv_encrypted_vector(this->context, io, cts);

        #ifdef HE_TIMING
        auto t1_preprocess = high_resolution_clock::now();
        #endif

        vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats(12);
        vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats_single(12);
        vector<vector<Plaintext>> bias_packing(12);

        for (int packing_index = 0; packing_index < 12; packing_index++) {
            vector<uint64_t *> matrix_mod_p1(common_dim);
            vector<uint64_t *> matrix_mod_p2(common_dim);
            vector<uint64_t *> matrix_mod_p3(common_dim);
            vector<uint64_t *> matrix_mod_p4(common_dim);
            vector<uint64_t *> matrix1(common_dim);
            vector<uint64_t *> matrix2(common_dim);
            for (int i = 0; i < common_dim; i++) {
                matrix_mod_p1[i] = new uint64_t[output_dim];
                matrix_mod_p2[i] = new uint64_t[output_dim];
                matrix_mod_p3[i] = new uint64_t[output_dim / 2];
                matrix_mod_p4[i] = new uint64_t[output_dim / 2];
                matrix1[i] = new uint64_t[output_dim];
                matrix2[i] = new uint64_t[output_dim];
                for (int j = 0; j < output_dim; j++) {
                    matrix_mod_p1[i][j] = neg_mod((int64_t)B1[packing_index][i][j], (int64_t)prime_mod);
                    matrix_mod_p2[i][j] = neg_mod((int64_t)B2[packing_index][i][j], (int64_t)prime_mod);
                    int64_t val = (int64_t)B1[packing_index][i][j];
                    if (val > int64_t(prime_mod / 2)) {
                        val = val - prime_mod;
                    }
                    matrix1[i][j] = val;
                    val = (int64_t)B2[packing_index][i][j];
                    if (val > int64_t(prime_mod / 2)) {
                        val = val - prime_mod;
                    }
                    matrix2[i][j] = val;
                }

                for (int j = 0; j < output_dim / 2; j++) {
                    matrix_mod_p3[i][j] = neg_mod((int64_t)B3[packing_index][i][j], (int64_t)prime_mod);
                    matrix_mod_p4[i][j] = neg_mod((int64_t)B3[packing_index][i][j + output_dim / 2], (int64_t)prime_mod);
                }
            }

            for (int i = 0; i < output_dim; i++) {
                Bias1[packing_index][i] = neg_mod((int64_t)Bias1[packing_index][i], (int64_t)prime_mod);
                Bias2[packing_index][i] = neg_mod((int64_t)Bias2[packing_index][i], (int64_t)prime_mod);
                Bias3[packing_index][i] = neg_mod((int64_t)Bias3[packing_index][i], (int64_t)prime_mod);
            }

            auto cross_mat = bert_cross_packing_matrix(matrix_mod_p1.data(), matrix_mod_p2.data(), data);
            auto cross_mat_single = bert_cross_packing_single_matrix(matrix_mod_p3.data(), matrix_mod_p4.data(), data);
            auto bias = bert_cross_packing_bias(Bias1[packing_index].data(), Bias2[packing_index].data(), Bias3[packing_index].data(), data);
            cross_mats[packing_index] = cross_mat;
            cross_mats_single[packing_index] = cross_mat_single;
            bias_packing[packing_index] = bias;
        }

        // print_pt(bias_packing[0][0], 8192);
        
        PRG128 prg;
        uint64_t *secret_share = new uint64_t[input_dim*output_dim];
        prg.random_mod_p<uint64_t>(secret_share, input_dim*output_dim, prime_mod);
        // auto encoded_mat1 = bert_efficient_preprocess_matrix(matrix_mod_p1.data(), data);
        // auto encoded_mat2 = bert_efficient_preprocess_matrix(matrix_mod_p2.data(), data);

        // auto rotation_masks = generate_rotation_masks(data);
        // auto cipher_masks = generate_cipher_masks(data);
        // auto depth3_masks = generate_depth3_masks(data);
        auto cross_masks = generate_cross_packing_masks(data);

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
        vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w / data.slot_count * 3 * 12);
        bert_cipher_plain_bsgs(cts, cross_mats, bias_packing, cross_mats_single, data, Cipher_plain_results);

        #ifdef HE_DEBUG
        auto temp_cipher_plain_res = bert_efficient_postprocess(Cipher_plain_results, 
        data);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 64; j++)
                cout << ((int64_t) temp_cipher_plain_res[i + j * 128] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
            cout << endl;
        }
        #endif

        #ifdef HE_TIMING
        auto t2_cipher_plain = high_resolution_clock::now();
        interval = (t2_cipher_plain - t1_cipher_plain)/1e+9;
        cout << "[Server] Cipher-Plaintext Matmul takes " << interval.count() << "sec" << endl;

        auto t1_cipher_cipher = high_resolution_clock::now();
        #endif 

        // auto HE_result = bert_efficient_cipher(data, Cipher_plain_results, rotation_masks, cipher_masks);
        // auto HE_result = bert_efficient_cipher_depth3(data, Cipher_plain_results, depth3_masks);

        // for (int i = 0; i < Cipher_plain_results.size(); i++)
        //     evaluator->mod_switch_to_next_inplace(Cipher_plain_results[i]);

        vector<Ciphertext> HE_result(3 * 12);
        bert_cipher_cipher_cross_packing(data, Cipher_plain_results, cross_masks, HE_result);

        int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;
        for (int i = 0; i < 12; i++) {
            HE_result[24 + i] = Cipher_plain_results[2 + i * packing_gap];
        }

        // HACK: for softmax - V
        vector<Ciphertext> V = {HE_result.begin() + data.image_size * data.image_size / data.slot_count * 12, HE_result.end()};

        // #pragma omp parallel for
        // for (int i = 0; i < HE_result.size(); i++) {
        //     evaluator->mod_switch_to_next_inplace(HE_result[i]);
        //     evaluator->mod_switch_to_next_inplace(HE_result[i]);
        // }

        #ifdef HE_TIMING
        auto t2_cipher_cipher = high_resolution_clock::now();
        interval = (t2_cipher_cipher - t1_cipher_cipher)/1e+9;
        cout << "[Server] Cipher-Cipher Matmul takes " << interval.count() << "sec" << endl;
        #endif 

        send_encrypted_vector(io, HE_result);

        cout << "[Server] Result sent" << endl;
        cout << "[Server] size of result (Bytes): " << io->counter - io_start << endl;

        // Below is computing softmax * V
        // HACK

        auto io_checkpoint = io->counter;

        uint64_t *softmax_S2 = new uint64_t[128 * 12 * 128];
        for (int i = 0; i < 128 * 12; i++) {
            for (int j = 0; j < 128; j++) {
                softmax_S2[i * 128 + j] = i * 1000 + j;
            }
        }
        auto soft_mask = softmax_mask(data);

        #ifdef HE_TIMING
        auto t1_softmax_v = high_resolution_clock::now();
        #endif

        vector<vector<vector<Plaintext>>> S2_pack = preprocess_softmax_s2(softmax_S2, data, soft_mask);

        vector<Ciphertext> S1_pack(data.image_size * data.image_size / data.slot_count * 12);
        recv_encrypted_vector(context, io, S1_pack);

        uint64_t *softmax_v_r = new uint64_t[128 * 12 * 64];
        for (int i = 0; i < 128 * 12; i++) {
            for (int j = 0; j < 64; j++) {
                softmax_v_r[i * 64 + j] = (i * 100 + j) % 1000;
            }
        }

        vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> R_pack = preprocess_softmax_v_r(softmax_v_r, data);

        vector<Ciphertext> softmax_V_result(12 * data.image_size * data.filter_w / data.slot_count);

        #ifdef HE_TIMING
        auto t1_softmax_v_computation = high_resolution_clock::now();
        #endif
        bert_softmax_V(S1_pack, S2_pack, V, R_pack, data, softmax_V_result);
        #ifdef HE_TIMING
        auto t2_softmax_v_computation = high_resolution_clock::now();
        interval = (t2_softmax_v_computation - t1_softmax_v_computation)/1e+9;
        cout << "[Server] Softmax - V Computation Time " << interval.count() << "sec" << endl;
        #endif
        
        send_encrypted_vector(io, softmax_V_result);

        cout << "[Server] Result sent" << endl;
        cout << "[Server] size of result (Bytes): " << io->counter - io_checkpoint << endl;

        // HACK
        // auto softmax_V = bert_postprocess_V(softmax_V_result, data, true);

        // for (int i = 1; i < 2; i++) {
        //     for (int j = 64; j < 128; j++)
        //         cout << ((int64_t) softmax_V[i + j * 128] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
        //     cout << endl;
        // }

        #ifdef HE_TIMING
        auto t2_softmax_v = high_resolution_clock::now();
        interval = (t2_softmax_v - t1_softmax_v)/1e+9;
        cout << "[Server] Softmax - V Total Time " << interval.count() << "sec" << endl;
        #endif

        // for (int i = 0; i < common_dim; i++) {
        //     delete[] matrix_mod_p1[i];
        //     delete[] matrix_mod_p2[i];
        //     delete[] matrix1[i];
        //     delete[] matrix2[i];
        // }
        delete[] secret_share;

        #ifdef HE_TIMING
        auto t2_total = high_resolution_clock::now();
        interval = (t2_total - t1_total)/1e+9;
        cout << "[Server] Total Time " << interval.count() << "sec" << endl;
        #endif 
    }
}