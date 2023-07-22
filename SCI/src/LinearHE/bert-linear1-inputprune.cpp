#include "LinearHE/bert-linear1-inputprune.h"
#include <omp.h>

using namespace std;
using namespace sci;
using namespace seal;

#define HE_TIMING
// #define HE_DEBUG

void PruneLin1Field::print_noise_budget_vec(vector<Ciphertext> v) {
    cout << "Noise budget: ";
    for(int i = 0; i < v.size(); i++){
        cout << YELLOW << decryptor->invariant_noise_budget(v[i]) << " ";
    }
    cout << RESET << endl;
}

void PruneLin1Field::print_ct(Ciphertext &ct, int len){
    Plaintext pt;
    decryptor->decrypt(ct, pt);
    print_pt(pt, len);
}

void PruneLin1Field::print_pt(Plaintext &pt, int len) {
    vector<uint64_t> dest(len, 0ULL);
    encoder->decode(pt, dest);
    cout << "Decode first 5 rows: ";
    int non_zero_count;
    for(int i = 0; i < len; i++){
        cout << dest[i] << " ";
        // if(dest[i] != 0){
        //     non_zero_count += 1;
        // }
    }
    // cout << "Non zero count: " << non_zero_count;
    cout << endl;
}

vector<Plaintext> PruneLin1Field::generate_cross_packing_masks(const FCMetadata &data) {
    vector<Plaintext> result(data.image_size * 2);
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++) {
        vector<uint64_t> mask1(data.slot_count, 0ULL);
        vector<uint64_t> mask2(data.slot_count, 0ULL);
        for (int k = 0; k < data.image_size - i; k++) {
            mask1[k + (i * data.image_size) % (data.slot_count / 2)] = 1;
            mask1[k + data.slot_count / 2 + (i * data.image_size) % (data.slot_count / 2)] = 1;
        }
        for (int k = data.image_size - i; k < data.image_size; k++) {
            mask2[k + (i * data.image_size) % (data.slot_count / 2)] = 1;
            mask2[k + data.slot_count / 2 + (i * data.image_size) % (data.slot_count / 2)] = 1;
        }
        Plaintext pt1, pt2;
        encoder->encode(mask1, pt1);
        encoder->encode(mask2, pt2);
        result[i] = pt1;
        result[i + data.image_size] = pt2;
    }
    return result;
}

vector<Ciphertext> PruneLin1Field::rotation_by_one_depth3(const FCMetadata &data, const Ciphertext &ct, int k) {

    int m = -(data.image_size - k);
    Ciphertext ct1;
    Ciphertext ct2;
    evaluator->rotate_rows(ct, k, *gal_keys, ct1);
    evaluator->rotate_rows(ct, m, *gal_keys, ct2);

    return {ct1, ct2};
}

// column-wise packing
vector<Ciphertext> PruneLin1Field::bert_efficient_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data) {
    vector<Ciphertext> cts((data.image_size * data.filter_h) / data.slot_count);
    
    #pragma omp parallel for
    for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
    {
        vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
        pod_matrix = vector<uint64_t>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
        Ciphertext ct;
        Plaintext pt;
        encoder->encode(pod_matrix, pt);
        encryptor->encrypt(pt, ct);
        cts[i] = ct;
    }
    return cts;
}

vector<vector<Plaintext>> PruneLin1Field::bert_cross_packing_bias(const vector<vector<uint64_t>> &bias, const FCMetadata &data) {
    vector<vector<Plaintext>> cross_bias_packing(6);
    int current_packing = 0, matrix1_pointer = 0, matrix2_pointer = 0;
    while(current_packing < 6) {
        vector<uint64_t> temp(data.slot_count, 0ULL);
        int next_flag = 0;
        int row = 0;
        if (matrix1_pointer == data.filter_w && matrix2_pointer == data.filter_w) {
            matrix1_pointer = 0;
            matrix2_pointer = 0;
            current_packing += 1;
            if (current_packing >= 6)
                break;
        }
        while (row < data.slot_count) {
            if (row >= data.slot_count / 2) {
                next_flag = 1;
            }
            if (next_flag == 0) {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = bias[current_packing * 2][matrix1_pointer];
                }
                matrix1_pointer++;
            }
            else {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = bias[current_packing * 2 + 1][matrix2_pointer];
                }
                matrix2_pointer++;
            }
            row += data.image_size;
        }
        Plaintext pt;
        encoder->encode(temp, pt);
        cross_bias_packing[current_packing].push_back(pt);
    }
    return cross_bias_packing;
}

vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> PruneLin1Field::bert_cross_packing_single_matrix(const vector<vector<vector<uint64_t>>> &weights, const FCMetadata &data) {
    vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> result(6);
    int num_diag = data.slot_count / data.image_size / 2;

    int n1 = 8;
    int n2 = 4;
    if (data.image_size == 64) {
        n1 = 16;
        n2 = 4;
    }

    int weight_height = data.filter_h;

    int num_matrix_per_row = weight_height / num_diag;
    int num_matrix_per_col = data.filter_w / num_diag;

    #pragma omp parallel for
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<uint64_t> temp2;
        vector<vector<Plaintext>> weightMatrix1;
        vector<vector<Plaintext>> weightMatrix2;
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            int matrix_flag = 0;
            for (int l = 0; l < num_diag; l++) {
                vector<Plaintext> temp_matrix_diag(weight_height * data.image_size / data.slot_count);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                            if (matrix_flag == 0)
                                temp2.push_back(weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                            else
                                temp2.push_back(weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
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
                vector<Plaintext> temp_matrix_diag(weight_height * data.image_size / data.slot_count);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                            if (matrix_flag == 0)
                                temp2.push_back(weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                            else
                                temp2.push_back(weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
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
        result[packing_ind] = std::make_pair(weightMatrix1, weightMatrix2);
    }
    return result;
}

vector<vector<vector<Plaintext>>> PruneLin1Field::bert_softmax_v_packing_single_matrix(const vector<vector<vector<uint64_t>>> &weights, const FCMetadata &data) {
    vector<vector<vector<Plaintext>>> result(6);
    int num_diag = data.slot_count / data.image_size / 2;

    int n1 = 4;
    int n2 = 8;
    if (data.image_size == 64) {
        n1 = 8;
        n2 = 8;
    }

    int weight_height = data.image_size;
    int num_matrix_per_row = weight_height / num_diag; // 1 or 4
    int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

    omp_set_nested(1);
    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<vector<Plaintext>> weightMatrix1(num_matrix_per_col * num_diag);
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            #pragma omp parallel for
            for (int l = 0; l < num_diag; l++) {
                vector<uint64_t> temp2, temp3;
                vector<Plaintext> temp_matrix_diag(num_matrix_per_row);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                                temp2.push_back(weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                                temp3.push_back(weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        }
                    }
                    std::rotate(temp2.begin(), temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                    std::rotate(temp3.begin(), temp3.begin() + temp3.size() - (l % n1) * data.image_size, temp3.end());
                    temp2.insert(temp2.end(), temp3.begin(), temp3.end());
                    Plaintext pt;
                    encoder->encode(temp2, pt);
                    temp_matrix_diag[matrix_diag_index] = pt;
                    matrix_diag_index++;
                    temp2.clear();
                    temp3.clear();
                }
                weightMatrix1[col_ind * num_diag + l] = temp_matrix_diag;
            }
        }
        result[packing_ind] = weightMatrix1;
    }
    return result;
}

void PruneLin1Field::bert_cipher_plain_bsgs(const vector<Ciphertext> &cts, 
                    const vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &wq_pack, 
                    const vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &wk_pack, 
                    const vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &wv_pack, 
                    const vector<vector<Plaintext>> &bq_pack, const vector<vector<Plaintext>> &bk_pack, 
                    const vector<vector<Plaintext>> &bv_pack, const FCMetadata &data, 
                    vector<Ciphertext> &result) {

    auto t1 = high_resolution_clock::now();
    vector<vector<Ciphertext>> rotatedIR(cts.size()); // cts.size() = 48
    int n1 = 8;
    int n2 = 4;
    if (data.image_size == 64) {
        n1 = 16;
        n2 = 4;
    }

    int num_diag = data.slot_count / data.image_size / 2;
    int num_matrix_per_col = data.filter_w / num_diag;
    cout << "[Server] Online Start" << endl;
    
    omp_set_nested(1);
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < cts.size(); i++)
    {   
        vector<Ciphertext> tmp(n1 * 2);
        tmp[0] = cts[i];

        #pragma omp parallel for
        for (int j = 1; j < n1; j++) {
            Ciphertext temp_rot;
            evaluator->rotate_rows(cts[i], (num_diag - j) * data.image_size, *gal_keys, temp_rot);
            tmp[j] = temp_rot;
        }

        #pragma omp parallel for
        for (int j = 0; j < n1; j++) {
            Ciphertext temp_rot;
            evaluator->rotate_columns(tmp[j], *gal_keys, temp_rot);
            tmp[j + n1] = temp_rot;
        }

        rotatedIR[i] = tmp;
        tmp.clear();
    }

    auto t2 = high_resolution_clock::now();
    auto ms_double = (t2 - t1)/1e+9;
    cout << "[Server] Online - rotation done " << ms_double.count() << endl;
    t1 = high_resolution_clock::now();

    vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w * 3 * 12 / data.slot_count, vector<Ciphertext>(n2));

    int temp_result_size = data.image_size * data.filter_w * 2 / data.slot_count;

    #pragma omp parallel for num_threads(2)
    for (int packing_index = 0; packing_index < 6; packing_index++) {
        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(temp_result_size * 3, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results_q(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Ciphertext>> temp_results_k(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Ciphertext>> temp_results_v(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Plaintext>> enc_weights_q1 = wq_pack[packing_index].first;
        vector<vector<Plaintext>> enc_weights_q2 = wq_pack[packing_index].second;
        vector<vector<Plaintext>> enc_weights_k1 = wk_pack[packing_index].first;
        vector<vector<Plaintext>> enc_weights_k2 = wk_pack[packing_index].second;
        vector<vector<Plaintext>> enc_weights_v1 = wv_pack[packing_index].first;
        vector<vector<Plaintext>> enc_weights_v2 = wv_pack[packing_index].second;

        #pragma omp parallel for
        // #pragma omp taskloop
        for (int k = 0; k < cts.size() * n2; k++) {
            int j = k / cts.size();
            int ct_i = k % cts.size();
            for (int l = 0; l < data.image_size * data.filter_w * 2 / data.slot_count; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct_l_q, ct_r_q, ct_l_k, ct_r_k, ct_l_v, ct_r_v;
                    evaluator->multiply_plain(rotatedIR[ct_i][i], enc_weights_q1[n1 * j + i + l * num_diag][ct_i], ct_l_q);
                    evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_weights_q2[n1 * j + i + l * num_diag][ct_i], ct_r_q);
                    evaluator->multiply_plain(rotatedIR[ct_i][i], enc_weights_k1[n1 * j + i + l * num_diag][ct_i], ct_l_k);
                    evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_weights_k2[n1 * j + i + l * num_diag][ct_i], ct_r_k);
                    evaluator->multiply_plain(rotatedIR[ct_i][i], enc_weights_v1[n1 * j + i + l * num_diag][ct_i], ct_l_v);
                    evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_weights_v2[n1 * j + i + l * num_diag][ct_i], ct_r_v);
                    if (i == 0) {
                        evaluator->add(ct_l_q, ct_r_q, temp_results_q[l][k]);
                        evaluator->add(ct_l_k, ct_r_k, temp_results_k[l][k]);
                        evaluator->add(ct_l_v, ct_r_v, temp_results_v[l][k]);
                    }
                    else {
                        Ciphertext temp_add_ct;
                        evaluator->add(ct_l_q, ct_r_q, temp_add_ct);
                        evaluator->add_inplace(temp_results_q[l][k], temp_add_ct);
                        evaluator->add(ct_l_k, ct_r_k, temp_add_ct);
                        evaluator->add_inplace(temp_results_k[l][k], temp_add_ct);
                        evaluator->add(ct_l_v, ct_r_v, temp_add_ct);
                        evaluator->add_inplace(temp_results_v[l][k], temp_add_ct);
                    }
                }
            }
        }

        #pragma omp parallel for
        // #pragma omp taskloop
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
                for (int l = 0; l < temp_result_size; l++) {
                    if (ct_i == 0) {
                        temp_results[l][j] = temp_results_q[l][j * cts.size() + ct_i];
                        temp_results[l + temp_result_size][j] = temp_results_k[l][j * cts.size() + ct_i];
                        temp_results[l + temp_result_size * 2][j] = temp_results_v[l][j * cts.size() + ct_i];
                    }
                    else {
                        evaluator->add_inplace(temp_results[l][j], temp_results_q[l][j * cts.size() + ct_i]);
                        evaluator->add_inplace(temp_results[l + temp_result_size][j], temp_results_k[l][j * cts.size() + ct_i]);
                        evaluator->add_inplace(temp_results[l + temp_result_size * 2][j], temp_results_v[l][j * cts.size() + ct_i]);
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int l = 0; l < temp_result_size; l++) {
            Ciphertext ct_q, ct_k, ct_v;
            for (int k = 0; k < n2; k++) {
                if (k == 0) {
                    ct_q = temp_results[l][0];
                    ct_k = temp_results[l + temp_result_size][0];
                    ct_v = temp_results[l + temp_result_size * 2][0];
                }
                else {
                    Ciphertext temp_rot_ct;
                    evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *gal_keys, temp_rot_ct);
                    evaluator->add_inplace(ct_q, temp_rot_ct);
                    evaluator->rotate_rows(temp_results[l + temp_result_size][k], -n1 * k * data.image_size, *gal_keys, temp_rot_ct);
                    evaluator->add_inplace(ct_k, temp_rot_ct);
                    evaluator->rotate_rows(temp_results[l + temp_result_size * 2][k], -n1 * k * data.image_size, *gal_keys, temp_rot_ct);
                    evaluator->add_inplace(ct_v, temp_rot_ct);
                }
            }
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count] = ct_q;
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count] = ct_k;
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 24 / data.slot_count] = ct_v;
            
            evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count], bq_pack[packing_index][l]);
            evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count], bk_pack[packing_index][l]);
            evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 24 / data.slot_count], bv_pack[packing_index][l]);
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

void PruneLin1Field::bert_cipher_cipher_cross_packing(const FCMetadata &data, const vector<Ciphertext> &Cipher_plain_result, const vector<Plaintext> &cross_masks, vector<Ciphertext> &results) {
    int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;
    int temp_result_size = data.image_size * data.filter_w * 2 / data.slot_count;

    #pragma omp parallel for num_threads(2)
    for (int packing_index = 0; packing_index < 6; packing_index++) {
        vector<Ciphertext> rotation_results(data.image_size * 2);
        for (int l = 0; l < temp_result_size; l++) {
            Ciphertext Qi = Cipher_plain_result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count];
            Ciphertext Ki = Cipher_plain_result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count];
            #pragma omp parallel for
            for (int i = 0; i < data.image_size; i++) {
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(data, Ki, i);
                if (l == 0) {
                    evaluator->multiply(Qi, temp_mult[0], rotation_results[i]);
                    evaluator->multiply(Qi, temp_mult[1], rotation_results[i + data.image_size]);
                }
                else {
                    Ciphertext temp_qk;
                    evaluator->multiply(Qi, temp_mult[0], temp_qk);
                    evaluator->add_inplace(rotation_results[i], temp_qk);
                    evaluator->multiply(Qi, temp_mult[1], temp_qk);
                    evaluator->add_inplace(rotation_results[i + data.image_size], temp_qk);
                }
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < data.image_size * 2; i++) {
            evaluator->relinearize_inplace(rotation_results[i], *relin_keys);
        }
        int local_rotation = std::ceil(std::log2(data.slot_count / data.image_size / 2));

        #pragma omp parallel for
        for (int i = 0; i < data.image_size * 2; i++) {
            for (int k = 0; k < local_rotation; k++) {
                Ciphertext temp2;
                evaluator->rotate_rows(rotation_results[i], (int32_t) pow(2, k) * data.image_size, *gal_keys, temp2);
                evaluator->add_inplace(rotation_results[i], temp2);
            }
            evaluator->multiply_plain_inplace(rotation_results[i], cross_masks[i]);
        }
        int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
        int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

        #pragma omp parallel for
        for (int i = 0; i < num_cts_per_res; i++) {
            results[packing_index * num_cts_per_res + i] = rotation_results[num_col_per_ct * i];
            for (int j = 1; j < num_col_per_ct; j++) {
                evaluator->add_inplace(results[packing_index * num_cts_per_res + i], rotation_results[num_col_per_ct * i + j]);
                evaluator->add_inplace(results[packing_index * num_cts_per_res + i], rotation_results[num_col_per_ct * i + j + data.image_size]);
            }
        }
    }
}

vector<vector<vector<uint64_t>>> PruneLin1Field::softmax_mask(const FCMetadata &data) {
    vector<vector<vector<uint64_t>>> mask(2, vector<vector<uint64_t>>(data.image_size, vector<uint64_t>(data.image_size)));
    #pragma omp parallel for
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
vector<vector<vector<Plaintext>>> PruneLin1Field::preprocess_softmax_s2(const uint64_t *matrix, const FCMetadata &data, vector<vector<vector<uint64_t>>> &mask) {

    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    vector<vector<vector<Plaintext>>> s2_pack(6);

    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<vector<Plaintext>> weightMatrix1(2, vector<Plaintext>(num_diag));
        #pragma omp parallel for
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            vector<uint64_t> temp2, temp3;
            vector<uint64_t> r1(data.image_size), r2(data.image_size), r3(data.image_size), r4(data.image_size);
            for (int j = 0; j < num_diag; j++) {
                temp2.push_back(matrix[((j + diag_ind) % num_diag) + j * data.image_size + packing_ind * 2 * data.image_size * data.image_size]);
                temp3.push_back(matrix[((j + diag_ind) % num_diag) + j * data.image_size + (packing_ind * 2 + 1) * data.image_size * data.image_size]);
            }
            // std::rotate(temp2.begin(), temp2.begin() + temp2.size() - diag_ind, temp2.end());
            std::transform(temp2.begin(), temp2.end(), mask[0][diag_ind].begin(), r1.begin(), std::multiplies<uint64_t>());
            std::transform(temp2.begin(), temp2.end(), mask[1][diag_ind].begin(), r2.begin(), std::multiplies<uint64_t>());
            std::transform(temp3.begin(), temp3.end(), mask[0][diag_ind].begin(), r3.begin(), std::multiplies<uint64_t>());
            std::transform(temp3.begin(), temp3.end(), mask[1][diag_ind].begin(), r4.begin(), std::multiplies<uint64_t>());
            for (int j = 0; j < std::log2(num_diag_per_ct); j++) {
                r1.reserve(r1.size() + distance(r1.begin(), r1.end()));
                r1.insert(r1.end(), r1.begin(), r1.end());
                r2.reserve(r2.size() + distance(r2.begin(), r2.end()));
                r2.insert(r2.end(), r2.begin(), r2.end());
                r3.reserve(r3.size() + distance(r3.begin(), r3.end()));
                r3.insert(r3.end(), r3.begin(), r3.end());
                r4.reserve(r4.size() + distance(r4.begin(), r4.end()));
                r4.insert(r4.end(), r4.begin(), r4.end());
            }
            r1.insert(r1.end(), r3.begin(), r3.end());
            r2.insert(r2.end(), r4.begin(), r4.end());

            Plaintext pt;
            encoder->encode(r1, pt);
            weightMatrix1[0][diag_ind] = pt;
            encoder->encode(r2, pt);
            weightMatrix1[1][diag_ind] = pt;
        }
        s2_pack[packing_ind] = weightMatrix1;
    }
    return s2_pack;
}

// matrix is row-packed with 12 * image_size rows and image_size cols
vector<Ciphertext> PruneLin1Field::preprocess_softmax_s1(const uint64_t *matrix, const FCMetadata &data) {
    int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
    int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

    int total_cts = 12 * data.image_size * data.image_size / data.slot_count;
    vector<Ciphertext> enc_softmax(total_cts);

    #pragma omp parallel for
    for (int ct_ind = 0; ct_ind < total_cts; ct_ind++) {
        int current_col = ct_ind % num_cts_per_res;
        int current_packing = ct_ind / num_cts_per_res;
        vector<uint64_t> pod_matrix(data.slot_count);

        for (int row = 0; row < data.slot_count; row++) {
            int j = row / data.image_size + current_col * num_col_per_ct;
            int k = row % data.image_size;
            int next_flag = 0;
            if (row >= data.slot_count / 2) {
                next_flag = data.image_size * data.image_size;
                j -= data.slot_count / 2 / data.image_size;
            }
            pod_matrix[row] = matrix[k * data.image_size + j + current_packing * data.image_size * data.image_size * 2 + next_flag];
        }
        Ciphertext ct;
        Plaintext pt;
        encoder->encode(pod_matrix, pt);
        encryptor->encrypt(pt, ct);
        enc_softmax[ct_ind] = ct;
    }

    #pragma omp parallel for
    for (int i = 0; i < enc_softmax.size(); i++) {
        evaluator->mod_switch_to_next_inplace(enc_softmax[i]);
    }
    return enc_softmax;
}


// matrix is col-packed with 128 rows and 64 * 12 cols
vector<vector<vector<Plaintext>>> PruneLin1Field::preprocess_softmax_v_r(const uint64_t *matrix, const FCMetadata &data) {
    vector<vector<vector<uint64_t>>> weights_r(12, vector<vector<uint64_t>>(data.image_size, vector<uint64_t>(data.filter_w)));

    for (int packing_ind = 0; packing_ind < 12; packing_ind++) {
        #pragma omp parallel for
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < data.filter_w; j++) {
                weights_r[packing_ind][i][j] = matrix[i + j * data.image_size + packing_ind * data.image_size * data.filter_w];
            }
        }
    }
    vector<vector<vector<Plaintext>>> R_pack = bert_softmax_v_packing_single_matrix(weights_r, data);
    return R_pack;
}

uint64_t* PruneLin1Field::client_S1_V_R(const uint64_t *softmax_s1, vector<Ciphertext> &V, const FCMetadata &data) {
    uint64_t* result = new uint64_t[12 * data.image_size * data.filter_w];
    // vector<vector<vector<uint64_t>>> result(12, vector<vector<uint64_t>> (data.image_size, vector<uint64_t> (data.filter_w)));
    auto V_R = bert_postprocess_V(V, data, true);
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        #pragma omp parallel for
        for(int i = 0; i < data.image_size; i++) {
            for(int j = 0; j < data.filter_w; j++) {
                result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = 0;
                for(int k = 0; k < data.image_size; k++) {
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] += neg_mod((int64_t)softmax_s1[packing_num * data.image_size * data.image_size + i * data.image_size + k] * V_R[k + j * data.image_size + data.image_size * data.filter_w * packing_num], (int64_t)prime_mod);
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = neg_mod((int64_t)result[packing_num * data.image_size * data.filter_w + i + j * data.image_size], (int64_t)prime_mod);
                }
            }
        }
    }
    return result;
}

void PruneLin1Field::bert_softmax_V(vector<Ciphertext> &softmax_s1, vector<vector<vector<Plaintext>>> &softmax_s2, vector<Ciphertext> &V, vector<vector<vector<Plaintext>>> &R, const FCMetadata &data, vector<Ciphertext> &result) {
    // FIXME: pack R according to ours ctxpt
    // FIXME: compute softmax_s1 x R

    #pragma omp parallel for
    for (int i = 0; i < V.size(); i++) {
        evaluator->mod_switch_to_next_inplace(V[i]);
    }
    int n1 = 4;
    int n2 = 8;
    if (data.image_size == 64) {
        n1 = 8;
        n2 = 8;
    }

    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        int num_diag = data.slot_count / data.image_size / 2;
        int num_matrix_per_row = data.image_size / num_diag; // 1 or 4
        int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

        vector<vector<Plaintext>> R1 = R[packing_ind];
        vector<vector<Ciphertext>> rotatedIR(num_matrix_per_row);

        #pragma omp parallel for
        for (int i = 0; i < num_matrix_per_row; i++) {
            vector<Ciphertext> tmp;
            tmp.push_back(softmax_s1[packing_ind * num_matrix_per_row + i]);

            for (int j = 1; j < n1; j++) {
                Ciphertext temp_rot;
                evaluator->rotate_rows(softmax_s1[packing_ind * num_matrix_per_row + i], (num_diag - j) * data.image_size, *gal_keys, temp_rot);
                tmp.push_back(temp_rot);
            }

            rotatedIR[i] = tmp;
            tmp.clear();
        }

        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(num_matrix_per_col, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results1(num_matrix_per_col, vector<Ciphertext>(n2 * num_matrix_per_row));

        #pragma omp parallel for
        for (int k = 0; k < num_matrix_per_row * n2; k++) {
            int j = k / num_matrix_per_row;
            int ct_i = k % num_matrix_per_row;
            for (int l = 0; l < num_matrix_per_col; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct1_l;
                    evaluator->multiply_plain(rotatedIR[ct_i][i], R1[n1 * j + i + l * num_diag][ct_i], ct1_l);
                    if (i == 0)
                        temp_results1[l][k] = ct1_l;
                    else {
                        evaluator->add_inplace(temp_results1[l][k], ct1_l);
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < num_matrix_per_row; ct_i++) {
                for (int l = 0; l < num_matrix_per_col; l++) {
                    if (ct_i == 0)
                        temp_results[l][j] = temp_results1[l][j * num_matrix_per_row + ct_i];
                    else
                        evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * num_matrix_per_row + ct_i]);
                }
            }
        }

        #pragma omp parallel for
        for (int l = 0; l < num_matrix_per_col; l++) {
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
            result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + l] = ct;
        }

        // FIXME: pack softmax_s2 according to gazelle
        // FIXME: compute softmax_s2 x V

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_matrix_per_col; ct_ind++) {
            vector<Ciphertext> rotation_results(num_diag);

            #pragma omp parallel for
            for (int i = 0; i < num_diag; i++) {
                Ciphertext temp1;
                Ciphertext temp2;
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(data, V[packing_ind * num_matrix_per_col + ct_ind], i);
                evaluator->multiply_plain(temp_mult[0], softmax_s2[packing_ind][0][i], temp1);
                evaluator->multiply_plain(temp_mult[1], softmax_s2[packing_ind][1][i], temp2);
                evaluator->add(temp1, temp2, rotation_results[i]);
            }
            for (int i = 0; i < num_diag; i++) {
                evaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind], rotation_results[i]);
            }
            rotation_results.clear();
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        evaluator->mod_switch_to_next_inplace(result[i]);
    }
}

uint64_t* PruneLin1Field::bert_postprocess_V(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing) {
    // uint64_t *result_V = new uint64_t[data.image_size*data.filter_w*12];
    uint64_t *result_V = new uint64_t[cts.size() * data.slot_count];
    // int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_V_per_cts = data.slot_count / (data.image_size * data.filter_w);

    omp_set_nested(1);
    #pragma omp parallel for
    for (int ct_ind = 0; ct_ind < cts.size(); ct_ind++) {
        vector<uint64_t> plain(data.slot_count, 0ULL);
        Plaintext pt;
        decryptor->decrypt(cts[ct_ind], pt);
        encoder->decode(pt, plain);
        if (col_packing) {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w;
                }
                if (num_V_per_cts == 1) {
                    result_V[k + j * data.image_size + (ct_ind / 2) * data.image_size * data.filter_w * 2 + (ct_ind % 2) * data.image_size * data.filter_w / 2] = plain[row];
                }
                else if (num_V_per_cts == 2) {
                    result_V[k + j * data.image_size + ct_ind * data.image_size * data.filter_w * 2] = plain[row];
                }
            }
        }
        else {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    next_flag = data.filter_w * data.image_size;
                }
                if (num_V_per_cts == 1) {
                    result_V[k * data.filter_w + j + next_flag + (ct_ind / 2) * data.image_size * data.filter_w * 2 + (ct_ind % 2) * data.filter_w / 2] = plain[row];
                }
                else if (num_V_per_cts == 2) {
                    result_V[k * data.filter_w + j + next_flag + ct_ind * data.image_size * data.filter_w * 2] = plain[row];
                }
            }
        }
    }
    return result_V;
}

uint64_t* PruneLin1Field::bert_cross_packing_postprocess(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing) {
    uint64_t *result_V = new uint64_t[data.image_size * data.image_size * 12];
   
    int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
    int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32
    
    omp_set_nested(1);
    #pragma omp parallel for
    for (int ct_ind = 0; ct_ind < cts.size(); ct_ind++) {
        vector<uint64_t> plain(data.slot_count, 0ULL);
        Plaintext pt;
        decryptor->decrypt(cts[ct_ind], pt);
        encoder->decode(pt, plain);
        int current_col = ct_ind % num_cts_per_res;
        int current_packing = ct_ind / num_cts_per_res;
        

        if (col_packing) {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size + current_col * num_col_per_ct;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    next_flag = data.image_size * data.image_size;
                    j -= data.slot_count / 2 / data.image_size;
                }
                result_V[k + (k + j) % data.image_size * data.image_size + current_packing * data.image_size * data.image_size * 2 + next_flag] = plain[row];
            }
        }
        else {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size + current_col * num_col_per_ct;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    next_flag = data.image_size * data.image_size;
                    j -= data.slot_count / 2 / data.image_size;
                }
                result_V[k * data.image_size + (k + j) % data.image_size + current_packing * data.image_size * data.image_size * 2 + next_flag] = plain[row];
            }
        }
    }
    return result_V;
}

PruneLin1Field::PruneLin1Field(int party, NetIO *io) {
    this->party = party;
    this->io = io;
    this->slot_count = 8192;

    this->party = party;
    this->io = io;
    generate_new_keys(party, io, slot_count, context, encryptor, decryptor,
                    evaluator, encoder, gal_keys, relin_keys, zero);
}

PruneLin1Field::~PruneLin1Field() {
    free_keys(party, encryptor, decryptor, evaluator, encoder, gal_keys, zero);
}

void PruneLin1Field::configure() {
  data.slot_count = 8192;
  // Only works with a ciphertext that fits in a single ciphertext
  assert(data.slot_count >= data.image_size);

  data.filter_size = data.filter_h * data.filter_w;
  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / next_pow2(data.filter_w);
  // How many total ciphertexts we'll need
  data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
}

void PruneLin1Field::matrix_multiplication(int32_t input_dim, 
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

        vector<Ciphertext> lin1_result(data.image_size * data.image_size * 12 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count);
        recv_encrypted_vector(context, io, lin1_result);

        vector<Ciphertext> ct_ct_result = {lin1_result.begin(), lin1_result.begin() + data.image_size * data.image_size * 12 / data.slot_count};
        
        vector<Ciphertext> V = {lin1_result.begin() + data.image_size * data.image_size * 12 / data.slot_count, lin1_result.end()};

        auto HE_result = bert_cross_packing_postprocess(ct_ct_result, data, false);

        // HACK: verify
        // cout << "col packing" << endl;
        // for (int i = 63; i < 64; i++) {
        //     for (int j = 0; j < data.image_size; j++)
        //         cout << ((int64_t) HE_result[i + j * data.image_size + data.image_size * data.image_size * 3] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
        //     cout << endl;
        // }

        // cout << "row packing" << endl;
        // for (int i = 63; i < 64; i++) {
        //     for (int j = 0; j < data.image_size; j++)
        //         cout << ((int64_t) HE_result[i * data.image_size + j + data.image_size * data.image_size * 3] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
        //     cout << endl;
        // }

        // HACK:
        // Below is computing softmax * V

        auto io_checkpoint = io->counter;

        uint64_t *softmax_s1 = new uint64_t[data.image_size * data.image_size * 12];
        for (int i = 0; i < data.image_size * 12; i++) {
            for (int j = 0; j < data.image_size; j++) {
                softmax_s1[i * data.image_size + j] = (i * 1000 + j);
            }
        }
        vector<Ciphertext> S1_pack = preprocess_softmax_s1(softmax_s1, data);

        auto S1_V_R = client_S1_V_R(softmax_s1, V, data);

        send_encrypted_vector(io, S1_pack);

        vector<Ciphertext> enc_softmax_V(12 * data.image_size * data.filter_w / data.slot_count);
        recv_encrypted_vector(context, io, enc_softmax_V);

        auto softmax_V = bert_postprocess_V(enc_softmax_V, data, true);

        for (int i = 0; i < data.image_size * data.filter_w * 12; i++) {
            softmax_V[i] += S1_V_R[i];
            softmax_V[i] = softmax_V[i] % prime_mod;
        }

        // cout << "[Client] Result sent" << endl;
        // cout << "[Client] size of result (Bytes): " << io->counter - io_checkpoint << endl;

        for (int i = 63; i < 64; i++) {
            for (int j = 0; j < data.filter_w; j++)
                cout << ((int64_t) softmax_V[i + j * data.image_size + data.image_size * data.filter_w * 3] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
            cout << endl;
        }

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
        vector<Ciphertext> cts(data.image_size * data.filter_h / data.slot_count);
        recv_encrypted_vector(this->context, io, cts);

        #ifdef HE_TIMING
        auto t1_preprocess = high_resolution_clock::now();
        #endif

        // vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats(12);
        // vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats_single(12);
        // vector<vector<Plaintext>> bias_packing(12);

        vector<vector<vector<uint64_t>>> weights_q(12, vector<vector<uint64_t>>(data.filter_h, vector<uint64_t>(data.filter_w)));
        vector<vector<vector<uint64_t>>> weights_k(12, vector<vector<uint64_t>>(data.filter_h, vector<uint64_t>(data.filter_w)));
        vector<vector<vector<uint64_t>>> weights_v(12, vector<vector<uint64_t>>(data.filter_h, vector<uint64_t>(data.filter_w)));

        vector<vector<uint64_t>> bias_q(12, vector<uint64_t>(data.filter_w));
        vector<vector<uint64_t>> bias_k(12, vector<uint64_t>(data.filter_w));
        vector<vector<uint64_t>> bias_v(12, vector<uint64_t>(data.filter_w));

        for (int packing_index = 0; packing_index < 12; packing_index++) {
            for (int i = 0; i < common_dim; i++) {
                for (int j = 0; j < output_dim; j++) {
                    weights_q[packing_index][i][j] = neg_mod((int64_t)B1[packing_index][i][j], (int64_t)prime_mod);
                    weights_k[packing_index][i][j] = neg_mod((int64_t)B2[packing_index][i][j], (int64_t)prime_mod);
                    weights_v[packing_index][i][j] = neg_mod((int64_t)B3[packing_index][i][j], (int64_t)prime_mod);
                }
            }
            for (int i = 0; i < output_dim; i++) {
                bias_q[packing_index][i] = neg_mod((int64_t)Bias1[packing_index][i], (int64_t)prime_mod);
                bias_k[packing_index][i] = neg_mod((int64_t)Bias2[packing_index][i], (int64_t)prime_mod);
                bias_v[packing_index][i] = neg_mod((int64_t)Bias3[packing_index][i], (int64_t)prime_mod);
            }
        }

        vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> wq_pack = bert_cross_packing_single_matrix(weights_q, data);
        vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> wk_pack = bert_cross_packing_single_matrix(weights_k, data);
        vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> wv_pack = bert_cross_packing_single_matrix(weights_v, data);

        vector<vector<Plaintext>> bq_pack = bert_cross_packing_bias(bias_q, data);
        vector<vector<Plaintext>> bk_pack = bert_cross_packing_bias(bias_k, data);
        vector<vector<Plaintext>> bv_pack = bert_cross_packing_bias(bias_v, data);

        auto cross_masks = generate_cross_packing_masks(data);

        #ifdef HE_TIMING
        auto t2_preprocess = high_resolution_clock::now();
        auto interval = (t2_preprocess - t1_preprocess)/1e+9;
        cout << "[Server] Preprocessing takes " << interval.count() << "sec" << endl;
        #endif

        #ifdef HE_TIMING
        auto t1_cipher_plain = high_resolution_clock::now();
        #endif 

        vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w * 3 * 12 / data.slot_count);
        bert_cipher_plain_bsgs(cts, wq_pack, wk_pack, wv_pack, bq_pack, bk_pack, bv_pack, data, Cipher_plain_results);

        #ifdef HE_TIMING
        auto t2_cipher_plain = high_resolution_clock::now();
        interval = (t2_cipher_plain - t1_cipher_plain)/1e+9;
        cout << "[Server] Cipher-Plaintext Matmul takes " << interval.count() << "sec" << endl;

        auto t1_cipher_cipher = high_resolution_clock::now();
        #endif 

        vector<Ciphertext> HE_result(data.image_size * data.image_size * 12 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count);
        bert_cipher_cipher_cross_packing(data, Cipher_plain_results, cross_masks, HE_result);

        for (int i = 0; i < data.image_size * data.filter_w * 12 / data.slot_count; i++) {
            HE_result[data.image_size * data.image_size * 12 / data.slot_count + i] = Cipher_plain_results[data.image_size * data.filter_w * 12 * 2 / data.slot_count + i];
        }

        #pragma omp parallel for
        for (int i = 0; i < HE_result.size(); i++) {
            evaluator->mod_switch_to_next_inplace(HE_result[i]);
            evaluator->mod_switch_to_next_inplace(HE_result[i]);
        }

        send_encrypted_vector(io, HE_result);

        // HACK: for softmax - V
        vector<Ciphertext> V = {Cipher_plain_results.begin() + data.image_size * data.filter_w * 12 * 2 / data.slot_count, Cipher_plain_results.end()};

        #ifdef HE_TIMING
        auto t2_cipher_cipher = high_resolution_clock::now();
        interval = (t2_cipher_cipher - t1_cipher_cipher)/1e+9;
        cout << "[Server] Cipher-Cipher Matmul takes " << interval.count() << "sec" << endl;
        #endif 

        cout << "[Server] Result sent" << endl;
        cout << "[Server] size of result (Bytes): " << io->counter - io_start << endl;

        // Below is computing softmax * V
        // HACK

        auto io_checkpoint = io->counter;

        uint64_t *softmax_S2 = new uint64_t[data.image_size * data.image_size * 12];
        for (int i = 0; i < data.image_size * 12; i++) {
            for (int j = 0; j < data.image_size; j++) {
                softmax_S2[i * data.image_size + j] = i * 1000 + j;
            }
        }
        auto soft_mask = softmax_mask(data);

        #ifdef HE_TIMING
        auto t1_softmax_v = high_resolution_clock::now();
        #endif

        vector<vector<vector<Plaintext>>> S2_pack = preprocess_softmax_s2(softmax_S2, data, soft_mask);

        vector<Ciphertext> S1_pack(data.image_size * data.image_size * 12 / data.slot_count);
        recv_encrypted_vector(context, io, S1_pack);

        uint64_t *softmax_v_r = new uint64_t[data.image_size * data.filter_w * 12];
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < data.filter_w * 12; j++) {
                softmax_v_r[i + j * data.image_size] = (i + j * 100);
            }
        }

        vector<vector<vector<Plaintext>>> R_pack = preprocess_softmax_v_r(softmax_v_r, data);

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
    }
}