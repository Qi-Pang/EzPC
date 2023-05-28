#include "linear.h"

Linear::Linear(){}

Linear::Linear(int party, NetIO *io) {
	this->party = party;
	this->io = io;
	this->he_8192 = new HE(
		party,
		io,
		8192,
		{54, 54, 55, 55},
		536903681
    );

	this->he_4096 = new HE(
		party,
		io,
		4096,
		{54, 55},
		65537
    );
}

Linear::~Linear() {

}


vector<Ciphertext> Linear::linear_1(
		HE* he,
		vector<Ciphertext> input_cts, 
		vector<vector<vector<uint64_t>>> w_q,
		vector<vector<vector<uint64_t>>> w_k,
		vector<vector<vector<uint64_t>>> w_v,
		vector<vector<uint64_t>> b_q,
		vector<vector<uint64_t>> b_k,
		vector<vector<uint64_t>> b_v,
		const FCMetadata &data) {

	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats(12);
	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats_single(12);
	vector<vector<Plaintext>> bias_packing(12);

	for (int packing_index = 0; packing_index < 12; packing_index++) {
		vector<uint64_t *> matrix_mod_p1(COMMON_DIM);
		vector<uint64_t *> matrix_mod_p2(COMMON_DIM);
		vector<uint64_t *> matrix_mod_p3(COMMON_DIM);
		vector<uint64_t *> matrix_mod_p4(COMMON_DIM);
		vector<uint64_t *> matrix1(COMMON_DIM);
		vector<uint64_t *> matrix2(COMMON_DIM);
		for (int i = 0; i < COMMON_DIM; i++) {
			matrix_mod_p1[i] = new uint64_t[OUTPUT_DIM];
			matrix_mod_p2[i] = new uint64_t[OUTPUT_DIM];
			matrix_mod_p3[i] = new uint64_t[OUTPUT_DIM / 2];
			matrix_mod_p4[i] = new uint64_t[OUTPUT_DIM / 2];
			matrix1[i] = new uint64_t[OUTPUT_DIM];
			matrix2[i] = new uint64_t[OUTPUT_DIM];
			for (int j = 0; j < OUTPUT_DIM; j++) {
				matrix_mod_p1[i][j] = neg_mod((int64_t)w_q[packing_index][i][j], (int64_t)prime_mod);
				matrix_mod_p2[i][j] = neg_mod((int64_t)w_k[packing_index][i][j], (int64_t)prime_mod);
				int64_t val = (int64_t)w_q[packing_index][i][j];
				if (val > int64_t(prime_mod / 2)) {
					val = val - prime_mod;
				}
				matrix1[i][j] = val;
				val = (int64_t)w_k[packing_index][i][j];
				if (val > int64_t(prime_mod / 2)) {
					val = val - prime_mod;
				}
				matrix2[i][j] = val;
			}

			for (int j = 0; j < OUTPUT_DIM / 2; j++) {
				matrix_mod_p3[i][j] = neg_mod((int64_t)w_v[packing_index][i][j], (int64_t)prime_mod);
				matrix_mod_p4[i][j] = neg_mod((int64_t)w_v[packing_index][i][j + OUTPUT_DIM / 2], (int64_t)prime_mod);
			}
		}

		for (int i = 0; i < OUTPUT_DIM; i++) {
			b_q[packing_index][i] = neg_mod((int64_t)b_q[packing_index][i], (int64_t)prime_mod);
			b_k[packing_index][i] = neg_mod((int64_t)b_k[packing_index][i], (int64_t)prime_mod);
			b_v[packing_index][i] = neg_mod((int64_t)b_v[packing_index][i], (int64_t)prime_mod);
		}

		auto cross_mat = bert_cross_packing_matrix(he, matrix_mod_p1.data(), matrix_mod_p2.data(), data);
		auto cross_mat_single = bert_cross_packing_single_matrix(he, matrix_mod_p3.data(), matrix_mod_p4.data(), data);
		auto bias = bert_cross_packing_bias(he, b_q[packing_index].data(), b_k[packing_index].data(), b_v[packing_index].data(), data);
		cross_mats[packing_index] = cross_mat;
		cross_mats_single[packing_index] = cross_mat_single;
		bias_packing[packing_index] = bias;
	}

	// PRG128 prg;
	// uint64_t *secret_share = new uint64_t[INPUT_DIM*OUTPUT_DIM];
	// prg.random_mod_p<uint64_t>(secret_share, INPUT_DIM*OUTPUT_DIM, prime_mod);


	auto cross_masks = generate_cross_packing_masks(he, data);


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

	vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w / data.slot_count * 3 * 12);
	bert_cipher_plain_bsgs(he, input_cts, cross_mats, bias_packing, cross_mats_single, data, Cipher_plain_results);

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

	vector<Ciphertext> HE_result(2 * 12);
	bert_cipher_cipher_cross_packing(he, data, Cipher_plain_results, cross_masks, HE_result);

	#pragma omp parallel for
	for (int i = 0; i < HE_result.size(); i++) {
		he->evaluator->mod_switch_to_next_inplace(HE_result[i]);
		he->evaluator->mod_switch_to_next_inplace(HE_result[i]);
	}

    return HE_result;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> 
Linear::bert_cross_packing_matrix(
	HE* he,
	const uint64_t *const *matrix1, 
	const uint64_t *const *matrix2, 
	const FCMetadata &data) {

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
                            he->encoder->encode(temp2, pt);
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
                            he->encoder->encode(temp2, pt);
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

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> 
Linear::bert_cross_packing_single_matrix(
	HE* he,
	const uint64_t *const *matrix1, 
	const uint64_t *const *matrix2, 
	const FCMetadata &data) {
		
    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    vector<uint64_t> temp2;
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = data.filter_h / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag / 2; // should be 8

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
                            he->encoder->encode(temp2, pt);
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
                            he->encoder->encode(temp2, pt);
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

vector<Plaintext> Linear::bert_cross_packing_bias(
	HE* he,
	const uint64_t *matrix1, 
	const uint64_t *matrix2, 
	const uint64_t *matrix3, 
	const FCMetadata &data) {

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
        he->encoder->encode(temp, pt);
        cross_bias_packing[packing_num] = pt;
        temp.clear();
    }
    int matrix3_pointer = 0;
    for (int packing_num = 2 * data.image_size * data.filter_w / data.slot_count; packing_num < 3 * data.image_size * data.filter_w / data.slot_count; packing_num++) {
        vector<uint64_t> temp(data.slot_count, 0ULL);
        int right_flag = 0;
        int row = 0;
        while (row < data.slot_count) {
            for (int i = 0; i < data.image_size; i++) {
                temp[row + i] = matrix3[matrix3_pointer];
            }
            row += data.image_size;
            matrix3_pointer++;
        }
        Plaintext pt;
        he->encoder->encode(temp, pt);
        cross_bias_packing[packing_num] = pt;
        temp.clear();
    }
    return cross_bias_packing;
}

vector<Plaintext> Linear::generate_cross_packing_masks(HE* he, const FCMetadata &data) {
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
        he->encoder->encode(mask1, pt);
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
        he->encoder->encode(mask1, pt);
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
        he->encoder->encode(mask1, pt);
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
        he->encoder->encode(mask1, pt);
        result.push_back(pt);
    }
    return result;
}

void Linear::bert_cipher_plain_bsgs(
	HE* he,
	const vector<Ciphertext> &cts, 
	const vector<pair<vector<vector<Plaintext>>, 
	vector<vector<Plaintext>>>> &cross_mats, 
	const vector<vector<Plaintext>> &Bias, 
	const vector<pair<vector<vector<Plaintext>>, 
	vector<vector<Plaintext>>>> &cross_mats_single, 
	const FCMetadata &data, 
	vector<Ciphertext> &result) {

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
    // cout << "[Server] Online Start" << endl;
    #pragma omp parallel for
    for (int i = 0; i < cts.size(); i++)
    {   
        vector<Ciphertext> tmp;
        tmp.push_back(cts[i]);

        for (int j = 1; j < n1; j++) {
            Ciphertext temp_rot;
            he->evaluator->rotate_rows(cts[i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
            tmp.push_back(temp_rot);
        }

        for (int j = 0; j < n1; j++) {
            Ciphertext temp_rot;
            he->evaluator->rotate_columns(tmp[j], *(he->gal_keys), temp_rot);
            tmp.push_back(temp_rot);
        }

        rotatedIR[i] = tmp;
        tmp.clear();
    }

    #ifdef HE_DEBUG
        cout << "[Server] Budget after rotation" << endl;
        print_noise_budget_vec(rotatedIR[0]);
    #endif

    // auto t2 = high_resolution_clock::now();
    // auto ms_double = (t2 - t1)/1e+9;
    // cout << "[Server] Online - rotation done " << ms_double.count() << endl;
    // t1 = high_resolution_clock::now();
    omp_set_nested(1);
    // #pragma omp parallel 
    // #pragma omp single
    #pragma omp parallel for num_threads(2)
    for (int packing_index = 0; packing_index < 12; packing_index++) {
        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w / data.slot_count * 3, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results1(data.image_size * data.filter_w / data.slot_count * 3, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Plaintext>> enc_mat1 = cross_mats[packing_index].first;
        vector<vector<Plaintext>> enc_mat2 = cross_mats[packing_index].second;
        vector<vector<Plaintext>> enc_mat3 = cross_mats_single[packing_index].first;
        vector<vector<Plaintext>> enc_mat4 = cross_mats_single[packing_index].second;

        #pragma omp parallel for num_threads(4)
        // #pragma omp taskloop
        for (int k = 0; k < cts.size() * n2; k++) {
            int j = k / cts.size();
            int ct_i = k % cts.size();
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count * 2; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct1_l;
                    Ciphertext ct1_r;
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], enc_mat1[n1 * j + i + l * num_diag][ct_i], ct1_l);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_mat2[n1 * j + i + l * num_diag][ct_i], ct1_r);
                    if (i == 0)
                        he->evaluator->add(ct1_l, ct1_r, temp_results1[l][k]);
                    else {
                        Ciphertext temp_add_ct;
                        he->evaluator->add(ct1_l, ct1_r, temp_add_ct);
                        he->evaluator->add_inplace(temp_results1[l][k], temp_add_ct);
                    }
                }
            }

            int third_index = data.image_size * data.filter_w / data.slot_count * 2;
            for (int l = third_index; l < data.image_size * data.filter_w / data.slot_count * 3; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct1_l;
                    Ciphertext ct1_r;
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], enc_mat3[n1 * j + i + (l - third_index) * num_diag][ct_i], ct1_l);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], enc_mat4[n1 * j + i + (l - third_index) * num_diag][ct_i], ct1_r);
                    if (i == 0)
                        he->evaluator->add(ct1_l, ct1_r, temp_results1[l][k]);
                    else {
                        Ciphertext temp_add_ct;
                        he->evaluator->add(ct1_l, ct1_r, temp_add_ct);
                        he->evaluator->add_inplace(temp_results1[l][k], temp_add_ct);
                    }
                }
            }
        }

        #pragma omp parallel for num_threads(4)
        // #pragma omp taskloop
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
                for (int l = 0; l < data.image_size * data.filter_w / data.slot_count * 3; l++) {
                    if (ct_i == 0)
                        temp_results[l][j] = temp_results1[l][j * cts.size() + ct_i];
                    else
                        he->evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * cts.size() + ct_i]);
                }
            }
            
        }

        #ifdef HE_DEBUG
            cout << "[Server] Budget after mult" << endl;
            print_noise_budget_vec(temp_results[0]);
        #endif

        #pragma omp parallel for num_threads(4)
        // #pragma omp taskloop
        for (int l = 0; l < data.image_size * data.filter_w / data.slot_count * 3; l++) {
            Ciphertext ct;
            for (int k = 0; k < n2; k++) {
                if (k == 0)
                    ct = temp_results[l][0];
                else {
                    Ciphertext temp_rot_ct;
                    he->evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct, temp_rot_ct);
                }
            }
            result[l + packing_index * data.image_size * data.filter_w / data.slot_count * 3] = ct;
            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w / data.slot_count * 3], Bias[packing_index][l]);
        }
    }

    // t2 = high_resolution_clock::now();
    // ms_double = (t2 - t1)/1e+9;
    // cout << "[Server] Online Done " << ms_double.count() << endl;

}

// 1. rotate rhs for 128 x 1-step rotations
// 2. mult with lhs (producing 128 cts)
// 3. for each of the 128 cts, rotate for log(32) times, sum together + 1 time batch rotation
// 4. mult masks (1, 0 (x31), 1, 0 (x31), ... ) and sum together (do the first 32 (1st batch) and then the second batch).

void Linear::bert_cipher_cipher_cross_packing(
	HE* he,
	const FCMetadata &data, 
	const vector<Ciphertext> &Cipher_plain_result, 
	const vector<Plaintext> &cross_masks, 
	vector<Ciphertext> &results) {
    int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;

    for (int packing_index = 0; packing_index < 12; packing_index++) {
        Ciphertext HE_result_1_left = Cipher_plain_result[0 + packing_gap * packing_index];
        Ciphertext HE_result_2_left = Cipher_plain_result[1 + packing_gap * packing_index];

        Ciphertext HE_result_1_right;
        Ciphertext HE_result_2_right;

        he->evaluator->rotate_columns(HE_result_1_left, *(he->gal_keys), HE_result_1_right);
        he->evaluator->rotate_columns(HE_result_2_left, *(he->gal_keys), HE_result_2_right);

        vector<Ciphertext> rotation_results(data.image_size + 2);
        auto t1 = high_resolution_clock::now();
        vector<Ciphertext> rotation_results_left(data.image_size + 2);
        vector<Ciphertext> rotation_results_right(data.image_size + 2);

        #pragma omp parallel for
        for (int i = 0; i <= data.image_size / 2; i++) {
            vector<Ciphertext> temp_mult = rotation_by_one_depth3(he, data, HE_result_1_right, i);

            he->evaluator->multiply(HE_result_1_left, temp_mult[0], rotation_results_left[i]);
            // evaluator->relinearize_inplace(rotation_results_left[i], *relin_keys);

            he->evaluator->multiply(HE_result_1_left, temp_mult[1], rotation_results_left[i + data.image_size / 2 + 1]);
            // evaluator->relinearize_inplace(rotation_results_left[i + data.image_size / 2 + 1], *relin_keys);

            temp_mult = rotation_by_one_depth3(he, data, HE_result_2_right, i);

            he->evaluator->multiply(HE_result_2_left, temp_mult[0], rotation_results_right[i]);
            // evaluator->relinearize_inplace(rotation_results_right[i], *relin_keys);

            he->evaluator->multiply(HE_result_2_left, temp_mult[1], rotation_results_right[i + data.image_size / 2 + 1]);
            // evaluator->relinearize_inplace(rotation_results_right[i + data.image_size / 2 + 1], *relin_keys);

            he->evaluator->add(rotation_results_left[i], rotation_results_right[i], rotation_results[i]);
            he->evaluator->relinearize_inplace(rotation_results[i], *(he->relin_keys));
            he->evaluator->add(rotation_results_left[i + data.image_size / 2 + 1], rotation_results_right[i + data.image_size / 2 + 1], rotation_results[i + data.image_size / 2 + 1]);
            he->evaluator->relinearize_inplace(rotation_results[i + data.image_size / 2 + 1], *(he->relin_keys));

        }
        auto t2 = high_resolution_clock::now();
        auto ms_double = (t2 - t1)/1e+9;
        // std::cout << "[Server] Cipher-Cipher Rotation 1 " << ms_double.count() << std::endl;

        t1 = high_resolution_clock::now();
        int local_rotation = std::ceil(std::log2(32));
        #pragma omp parallel for
        for (int i = 0; i < data.image_size + 2; i++) {
            for (int k = 0; k < local_rotation; k++) {
                Ciphertext temp2;
                he->evaluator->rotate_rows(rotation_results[i], (int32_t) pow(2, k) * 128, *(he->gal_keys), temp2);
                he->evaluator->add_inplace(rotation_results[i], temp2);
            }
            he->evaluator->multiply_plain_inplace(rotation_results[i], cross_masks[i]);
        }
        t2 = high_resolution_clock::now();
        ms_double = (t2 - t1)/1e+9;
        // std::cout << "[Server] Cipher-Cipher Rotation 2 " << ms_double.count() << std::endl;
        // Packing
        t1 = high_resolution_clock::now();
        
        he->evaluator->add(rotation_results[0], rotation_results[65], results[0 + 2 * packing_index]);
        he->evaluator->add(rotation_results[32], rotation_results[32 + 65], results[1 + 2 * packing_index]);

        for (int i = 1; i < 32; i++) {
            Ciphertext temp;
            he->evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[i]);
            he->evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[i + 65]);
            he->evaluator->add_inplace(results[1 + 2 * packing_index], rotation_results[i + 32]);
            he->evaluator->add_inplace(results[1 + 2 * packing_index], rotation_results[i + 32 + 65]);
        }

        he->evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[64]);
        he->evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[64 + 65]);
        t2 = high_resolution_clock::now();
        ms_double = (t2 - t1)/1e+9;
        // std::cout << "[Server] Cipher-Cipher Packing " << ms_double.count() << std::endl;
    }
}

vector<Ciphertext> Linear::rotation_by_one_depth3(
	HE* he,
	const FCMetadata &data, 
	const Ciphertext &ct, 
	int k) {

    int m = -(128 - k);
    Ciphertext ct1;
    Ciphertext ct2;
    he->evaluator->rotate_rows(ct, k, *(he->gal_keys), ct1);
    he->evaluator->rotate_rows(ct, m, *(he->gal_keys), ct2);

    return {ct1, ct2};
}

// column-wise packing
vector<Ciphertext> Linear::bert_efficient_preprocess_vec(
	HE* he,
	vector<uint64_t> &input, 
	const FCMetadata &data) {

    vector<int64_t> pod_matrix(data.slot_count, 0ULL);
    vector<Ciphertext> cts;
    for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
    {
        pod_matrix = vector<int64_t>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
        Ciphertext ct;
        Plaintext pt;
        he->encoder->encode(pod_matrix, pt);
        he->encryptor->encrypt(pt, ct);
        cts.push_back(ct);
    }
    return cts;
}

