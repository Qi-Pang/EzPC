#include "linear.h"


void print_pt_l(HE* he, Plaintext &pt, int len) {
    vector<int64_t> dest(len, 0ULL);
    he->encoder->decode(pt, dest);
    cout << "Decode first 5 rows: ";
    int non_zero_count;
    for(int i = 0; i < 10; i++){
        cout << dest[i] << " ";
        // if(dest[i] != 0){
        //     non_zero_count += 1;
        // }
    }
    // cout << "Non zero count: " << non_zero_count;
    cout << endl;
}

void print_ct_l(HE* he, Ciphertext &ct, int len){
    Plaintext pt;
    he->decryptor->decrypt(ct, pt);
    cout << "Noise budget: ";
    cout << YELLOW << he->decryptor->invariant_noise_budget(ct) << " ";
    cout << RESET << endl;
    print_pt_l(he, pt, len);
}

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

    // this->he_8192_tiny = new HE(
    //     party,
    //     io,
    //     8192,
    //     {54, 54, 55, 55},
	// 	536903681
    // );

    this->he_8192_tiny = new HE(
        party,
        io,
        8192,
        {60, 60},
		557057
    );

    this->p_mod = prime_mod;

	// this->he_4096 = new HE(
	// 	party,
	// 	io,
	// 	4096,
	// 	{54, 55},
	// 	65537
    // );

    pp_1.resize(ATTENTION_LAYERS);
    pp_2.resize(ATTENTION_LAYERS);
    pp_3.resize(ATTENTION_LAYERS);
    pp_4.resize(ATTENTION_LAYERS);

    data_lin1.filter_h = COMMON_DIM;
    data_lin1.filter_w = OUTPUT_DIM;
    data_lin1.image_size = INPUT_DIM;
    data_lin1.slot_count = 8192;

    data_lin2.filter_h = COMMON_DIM;
    data_lin2.filter_w = COMMON_DIM;
    data_lin2.image_size = INPUT_DIM;
    data_lin2.slot_count = 8192;

    data_lin3.filter_h = COMMON_DIM;
    data_lin3.filter_w = INTER_DIM;
    data_lin3.image_size = INPUT_DIM;
    data_lin3.slot_count = 8192;

    data_lin4.filter_h = INTER_DIM;
    data_lin4.filter_w = COMMON_DIM;
    data_lin4.image_size = INPUT_DIM;
    data_lin4.slot_count = 8192;
}

Linear::~Linear() {

}

PreprocessParams_1 Linear::params_preprocessing_ct_ct(
    HE* he,
    vector<vector<vector<uint64_t>>> w_q,
    vector<vector<vector<uint64_t>>> w_k,
    vector<vector<vector<uint64_t>>> w_v,
    vector<vector<uint64_t>> b_q,
    vector<vector<uint64_t>> b_k,
    vector<vector<uint64_t>> b_v,
    const FCMetadata &data
){
    PreprocessParams_1 pp;

    uint64_t plain_mod = he->plain_mod;

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
				matrix_mod_p1[i][j] = neg_mod((int64_t)w_q[packing_index][i][j], (int64_t)plain_mod);
				matrix_mod_p2[i][j] = neg_mod((int64_t)w_k[packing_index][i][j], (int64_t)plain_mod);
				int64_t val = (int64_t)w_q[packing_index][i][j];
				if (val > int64_t(plain_mod / 2)) {
					val = val - plain_mod;
				}
				matrix1[i][j] = val;
				val = (int64_t)w_k[packing_index][i][j];
				if (val > int64_t(plain_mod / 2)) {
					val = val - plain_mod;
				}
				matrix2[i][j] = val;
			}

			for (int j = 0; j < OUTPUT_DIM / 2; j++) {
				matrix_mod_p3[i][j] = neg_mod((int64_t)w_v[packing_index][i][j], (int64_t)plain_mod);
				matrix_mod_p4[i][j] = neg_mod((int64_t)w_v[packing_index][i][j + OUTPUT_DIM / 2], (int64_t)plain_mod);
			}
		}

		for (int i = 0; i < OUTPUT_DIM; i++) {
			b_q[packing_index][i] = neg_mod((int64_t)b_q[packing_index][i], (int64_t)plain_mod);
			b_k[packing_index][i] = neg_mod((int64_t)b_k[packing_index][i], (int64_t)plain_mod);
			b_v[packing_index][i] = neg_mod((int64_t)b_v[packing_index][i], (int64_t)plain_mod);
		}

		auto cross_mat = bert_cross_packing_matrix(he, matrix_mod_p1.data(), matrix_mod_p2.data(), data);
		auto cross_mat_single = bert_cross_packing_single_matrix(he, matrix_mod_p3.data(), matrix_mod_p4.data(), data, false);
		auto bias = bert_cross_packing_bias(he, b_q[packing_index].data(), b_k[packing_index].data(), b_v[packing_index].data(), data);
		pp.cross_mats.push_back(cross_mat);
		pp.cross_mats_single.push_back(cross_mat_single);
		pp.bias_packing.push_back(bias);
	}
    // print_pt_l(he_8192, bias_packing[0][0], 8192);

	pp.cross_masks = generate_cross_packing_masks(he, data);

    return pp;
}

PreprocessParams_2 Linear::params_preprocessing_ct_pt(
    HE* he,
    int32_t input_dim, 
    int32_t common_dim, 
    int32_t output_dim,
    vector<vector<uint64_t>> w,
    vector<uint64_t> b,
    const FCMetadata &data
){
    PreprocessParams_2 pp;
    uint64_t plain_mod = he->plain_mod;

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
            matrix_mod_p1[i][j] = neg_mod((int64_t)w[i][j], (int64_t)plain_mod);
            matrix_mod_p2[i][j] = neg_mod((int64_t)w[i][j + output_dim / 2], (int64_t)plain_mod);
        }
    }
    for (int i = 0; i < output_dim; i++) {
        b[i] = neg_mod((int64_t)b[i], (int64_t)plain_mod);
    }
    pp.cross_mat_single = bert_cross_packing_single_matrix_2(he, matrix_mod_p1.data(), matrix_mod_p2.data(), data);
    pp.cross_bias_single = bert_cross_packing_bias_2(he, b.data(), data);
    return pp;
}

void Linear::weights_preprocess(BertModel &bm){
    #pragma omp parallel for
    for(int i = 0; i < ATTENTION_LAYERS; i++){
        pp_1[i] = params_preprocessing_ct_ct(
            he_8192,
            bm.w_q[i],
            bm.w_k[i],
            bm.w_v[i],
            bm.b_q[i],
            bm.b_k[i],
            bm.b_v[i],
            data_lin1
        );

        pp_2[i] = params_preprocessing_ct_pt(
            he_8192_tiny,
            INPUT_DIM,
            COMMON_DIM,
            COMMON_DIM,
            bm.w_o[i],
            bm.b_o[i],
            data_lin2
        );

        pp_3[i] = params_preprocessing_ct_pt(
            he_8192_tiny,
            INPUT_DIM,
            COMMON_DIM,
            INTER_DIM,
            bm.w_i_1[i],
            bm.b_i_1[i],
            data_lin3
        );

        pp_4[i] = params_preprocessing_ct_pt(
            he_8192_tiny,
            INPUT_DIM,
            INTER_DIM,
            COMMON_DIM,
            bm.w_i_2[i],
            bm.b_i_2[i],
            data_lin4
        );
    }

    w_ln_1 = bm.w_ln_1;
    b_ln_1 = bm.b_ln_1;

    w_ln_2 = bm.w_ln_2;
    b_ln_2 = bm.b_ln_2;

    w_c = bm.w_c;
    b_c = bm.b_c;
    w_p = bm.w_p;
    b_p = bm.b_p;
}

vector<Ciphertext> Linear::linear_1(
		HE* he,
		vector<Ciphertext> input_cts, 
		PreprocessParams_1 &pp,
		const FCMetadata &data) {


	vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w / data.slot_count * 3 * 12);
	bert_cipher_plain_bsgs(he, input_cts, pp.cross_mats, pp.bias_packing, pp.cross_mats_single, data, Cipher_plain_results);

	vector<Ciphertext> HE_result(3 * 12);
	bert_cipher_cipher_cross_packing(he, data, Cipher_plain_results, pp.cross_masks, HE_result);

    int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;
    for (int i = 0; i < 12; i++) {
        HE_result[24 + i] = Cipher_plain_results[2 + i * packing_gap];
    }

    return HE_result;
}

vector<Ciphertext> Linear::linear_2(
    HE* he,
    vector<Ciphertext> input_cts, 
    PreprocessParams_2 &pp,
    const FCMetadata &data
    ){
    vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w / data.slot_count);
    bert_cipher_plain_bsgs_2(he, input_cts, pp.cross_mat_single.first, pp.cross_mat_single.second, pp.cross_bias_single, data, Cipher_plain_results);

    return Cipher_plain_results;
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
	const FCMetadata &data,
    const bool &softmax_V_flag) {
		
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
Linear::bert_cross_packing_single_matrix_2(
    HE* he,
    const uint64_t *const *matrix1,
    const uint64_t *const *matrix2,
    const FCMetadata &data){
    
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
        he->encoder->encode(temp, pt);
        cross_bias_packing[packing_num] = pt;
        temp.clear();
    }
    return cross_bias_packing;
}

vector<Plaintext> Linear::bert_cross_packing_bias_2(
    HE* he,
    const uint64_t *matrix, 
    const FCMetadata &data){

    vector<Plaintext> cross_bias_packing(data.image_size * data.filter_w / data.slot_count);
    int matrix_pointer1 = 0;
    int matrix_pointer2 = data.filter_w / 2;
    for (int packing_num = 0; packing_num < data.image_size * data.filter_w / data.slot_count; packing_num++) {
        vector<uint64_t> temp(data.slot_count, 0ULL);
        int right_flag = 0;
        int row = 0;
        while (row < data.slot_count) {
            if (row < data.slot_count / 2) {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = matrix[matrix_pointer1];
                }
                matrix_pointer1++;
            }
            else {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = matrix[matrix_pointer2];
                }
                matrix_pointer2++;
            }
            row += data.image_size;
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

    // auto t1 = high_resolution_clock::now();
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

    omp_set_nested(1);

    #pragma omp parallel for num_threads(12)
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
            #pragma omp parallel for num_threads(4)
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

    // auto t2 = high_resolution_clock::now();
    // auto ms_double = (t2 - t1)/1e+9;
    // cout << "[Server] Online Done " << ms_double.count() << endl;

}

void Linear::bert_cipher_plain_bsgs_2(
    HE* he,
    const vector<Ciphertext> &cts, 
    const vector<vector<Plaintext>> &enc_mat1, 
    const vector<vector<Plaintext>> &enc_mat2, 
    const vector<Plaintext> &enc_bias, 
    const FCMetadata &data, 
    vector<Ciphertext> &result){
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

    //compute matrix multiplication
    vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2));
    vector<vector<Ciphertext>> temp_results1(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2 * cts.size()));

    // rotatedIR 48 x 16, enc_mat 64 x 48

    #pragma omp parallel for
    for (int k = 0; k < cts.size() * n2; k++) {
        int j = k / cts.size();
        int ct_i = k % cts.size();
        for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
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
    }

    #pragma omp parallel for
    for (int j = 0; j < n2; j++) {
        for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                if (ct_i == 0)
                    temp_results[l][j] = temp_results1[l][j * cts.size() + ct_i];
                else
                    he->evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * cts.size() + ct_i]);
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
                he->evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                he->evaluator->add_inplace(ct, temp_rot_ct);
            }
        }
        result[l] = ct;
        he->evaluator->add_plain_inplace(result[l], enc_bias[l]);
    }
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

    vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
    vector<Ciphertext> cts;
    for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
    {
        pod_matrix = vector<uint64_t>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
        Ciphertext ct;
        Plaintext pt;
        he->encoder->encode(pod_matrix, pt);
        he->encryptor->encrypt(pt, ct);
        cts.push_back(ct);
    }
    return cts;
}

uint64_t* Linear::bert_cross_packing_postprocess(
    HE* he,
    vector<Ciphertext> &cts, 
    const FCMetadata &data) {
    uint64_t *result = new uint64_t[data.image_size*data.image_size*12];
    int num_cts_per_mat = data.image_size * data.image_size / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat; i++) {
            vector<uint64_t> plain(data.slot_count, 0ULL);
            Plaintext tmp;
            he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat], tmp);
            he->encoder->decode(tmp, plain);

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
                    result[k * data.image_size + (k + j - 32 + i * 32) % 128 + packing_num * data.image_size * data.image_size] = plain[row];
                }
            }
        }
    }
    return result;
}


void Linear::plain_cross_packing_postprocess(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data) {
    int num_cts_per_mat = data.image_size * data.image_size / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat; i++) {
            int offset = i + packing_num * num_cts_per_mat;
            vector<uint64_t> plain(&input[offset* data.slot_count], &input[offset* data.slot_count + data.slot_count]);

            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (j < 32) { // k, (k + j) % 128
                        output[k + ((k + j + i * 32) % data.image_size) * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                    else if (j == 32 && i == 0) { // (64 + k) % 128, k
                        output[((k + 64) % data.image_size) + k * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                    else { // (k - 32 + j) % 128, k
                        output[k * data.image_size + (k + j - 32 + i * 32) % data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                }
            }
            else {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (j < 32) { // k, (k + j) % 128
                        output[k * data.image_size + ((k + j + i * 32) % data.image_size) + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                    else if (j == 32 && i == 0) { // (64 + k) % 128, k
                        output[((k + 64) % data.image_size) * data.image_size + k + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                    else { // (k - 32 + j) % 128, k
                        output[k + ((k + j - 32 + i * 32) % data.image_size) * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                    }
                }
            }
        }
    }
}

void Linear::plain_cross_packing_postprocess_v(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data) {
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    int num_cts_per_mat = data.image_size * data.image_size / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            int offset = i + packing_num * num_cts_per_mat_V;
            vector<uint64_t> plain(&input[offset* data.slot_count], &input[offset* data.slot_count + data.slot_count]);
            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    output[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
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
                    output[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
}

void Linear::plain_col_packing_preprocess(
    uint64_t* input, 
    uint64_t * output,
    uint64_t plain_mod,
    int input_dim,
    int common_dim){
    for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                output[j*input_dim + i] = input[i*common_dim +j];
}

void Linear::plain_col_packing_preprocess_vec(
    vector<vector<uint64_t>> input, 
    uint64_t * output,
    uint64_t plain_mod,
    int input_dim,
    int common_dim){
    for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                output[j*input_dim + i] = input[i][j];
}

void Linear::plain_col_packing_postprocess(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data){
    for (int i = 0; i < data.image_size * data.filter_w / data.slot_count; i++) {
        int offset = i*data.slot_count;
        if (col_packing) {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                // int j = row / data.image_size + i * data.slot_count / data.image_size;
                // int k = row % data.image_size;
                // output[k + j * data.image_size] = input[row + offset];
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w / 2;
                }
                output[k + j * data.image_size + i * data.slot_count / 2] = input[row+ offset];
            }
        }
        else {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                // int j = row / data.image_size + i * data.slot_count / data.image_size;
                // int k = row % data.image_size;
                // k = k + j / data.filter_w * data.image_size;
                // j = j % data.filter_w;
                // output[k * data.filter_w + j] = input[row + offset];
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w / 2;
                }
                j += i * data.slot_count / data.image_size / 2;
                output[k * data.filter_w + j] = input[row+ offset];
            }
        }
    }
}

vector<vector<uint64_t>> Linear::concat_vec(
    uint64_t* att,
    int n,
    int dim1,
    int dim2){
    
    vector<vector<uint64_t>> res;
    for(int j = 0; j < dim1; j++){
        vector<uint64_t> row;
        for(int i = 0; i < n; i++){
            row.insert(row.end(), &att[i*dim1*dim2 + j*dim2], &att[i*dim1*dim2 + j*dim2 + dim2]);
        }
        res.push_back(row);
    }
    return res;
}

void Linear::concat( 
    uint64_t* input,
    uint64_t* output,
    int n,
    int dim1,
    int dim2){

    for(int j = 0; j < dim1; j++){
        for(int i = 0; i < n; i++){
            memcpy(&output[j*n*dim2 + i*dim2], &input[i*dim1*dim2 + j*dim2], dim2*sizeof(uint64_t));
        }
    }
}

// matrix is row-packed with 12 * 128 rows and 128 cols
vector<Ciphertext> Linear::preprocess_softmax_s1(HE* he, uint64_t* matrix, const FCMetadata &data) {
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
            he->encoder->encode(pod_matrix, pt);
            he->encryptor->encrypt(pt, ct);
            enc_softmax[packing_ind * num_ct_per_matrix + k] = ct;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < enc_softmax.size(); i++) {
        he->evaluator->mod_switch_to_next_inplace(enc_softmax[i]);
    }
    return enc_softmax;
}

void Linear::client_S1_V_R(HE* he, const uint64_t *softmax_s1, uint64_t* V, uint64_t* result, const FCMetadata &data){
    int total_packing = 12;
    uint64_t plain_mod = he->plain_mod;
    for (int packing_num = 0; packing_num < total_packing; packing_num++) {
        #pragma omp parallel for
        for(int i = 0; i < data.image_size; i++) {
            for(int j = 0; j < data.filter_w; j++) {
                result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = 0;
                for(int k = 0; k < data.image_size; k++) {
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] += neg_mod((int64_t)softmax_s1[(packing_num * data.image_size + i) * data.image_size + k] * V[k + j * data.image_size + data.image_size * data.filter_w * packing_num], (int64_t)plain_mod);
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = neg_mod((int64_t)result[packing_num * data.image_size * data.filter_w + i + j * data.image_size], (int64_t)plain_mod);
                }
            }
        }
    }
}

void Linear::bert_postprocess_V(HE* he, uint64_t* input, uint64_t* result, const FCMetadata &data, const bool &col_packing){
    // int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            // vector<uint64_t> plain(data.slot_count, 0ULL);
            // Plaintext pt;
            // he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat_V], pt);
            // he->encoder->decode(pt, plain);

            int offset = i + packing_num * num_cts_per_mat_V;
            vector<uint64_t> plain(&input[offset* data.slot_count], &input[offset* data.slot_count + data.slot_count]);

            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    result[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
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
                    result[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
}

void Linear::bert_postprocess_V_enc(HE* he, vector<Ciphertext> cts, uint64_t* result, const FCMetadata &data, const bool &col_packing){
    // int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            vector<uint64_t> plain(data.slot_count, 0ULL);
            Plaintext pt;
            he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat_V], pt);
            he->encoder->decode(pt, plain);

            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    result[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
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
                    result[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
}

vector<vector<vector<uint64_t>>> softmax_mask(const FCMetadata &data) {
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

vector<vector<vector<Plaintext>>> 
Linear::preprocess_softmax_s2(HE* he, const uint64_t *matrix, const FCMetadata &data){
    
    auto mask = softmax_mask(data);

    uint64_t *new_matrix = new uint64_t[128 * 12 * 128];
    for (int packing_ind = 0; packing_ind < 12; packing_ind++) {
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < 128; j++) {
                new_matrix[i * 128 + j + 128 * 128 * packing_ind] = matrix[j * 128 + i + 128 * 128 * packing_ind];
            }
        }
    }
    matrix = new_matrix;

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
            he->encoder->encode(r1, pt);
            weightMatrix1[0][diag_ind] = pt;
            he->encoder->encode(r2, pt);
            weightMatrix1[1][diag_ind] = pt;
            temp2.clear();
            r1.clear();
            r2.clear();
        }
        s2_pack[packing_ind] = weightMatrix1;
    }
    return s2_pack;
}

vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> 
Linear::preprocess_softmax_v_r(HE* he, const uint64_t *matrix, const FCMetadata &data) {
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
        auto temp_R_pack = bert_cross_packing_single_matrix(he, matrix_1.data(), matrix_2.data(), data, true);
        R_pack[packing_ind] = temp_R_pack;
    }
    return R_pack;
}

void Linear::bert_softmax_V(HE* he, vector<Ciphertext> &softmax_s1, vector<vector<vector<Plaintext>>> &softmax_s2, vector<Ciphertext> &V, vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> &R, const FCMetadata &data, vector<Ciphertext> &result) {
    // FIXME: pack R according to ours ctxpt
    // FIXME: compute softmax_s1 x R

    #pragma omp parallel for
    for (int i = 0; i < V.size(); i++) {
        he->evaluator->mod_switch_to_next_inplace(V[i]);
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
                he->evaluator->rotate_rows(softmax_s1[packing_ind * num_ct_per_s1 + i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
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

        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results1(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2 * num_ct_per_s1));

        #pragma omp parallel for
        for (int k = 0; k < num_ct_per_s1 * n2; k++) {
            int j = k / num_ct_per_s1;
            int ct_i = k % num_ct_per_s1;
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct1_l;
                    Ciphertext ct1_r;
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], R1[n1 * j + i + l * num_diag][ct_i], ct1_l);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], R2[n1 * j + i + l * num_diag][ct_i], ct1_r);
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

        #pragma omp parallel for
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < num_ct_per_s1; ct_i++) {
                for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                    if (ct_i == 0)
                        temp_results[l][j] = temp_results1[l][j * num_ct_per_s1 + ct_i];
                    else
                        he->evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * num_ct_per_s1 + ct_i]);
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
                    he->evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct, temp_rot_ct);
                }
            }
            result[packing_ind * data.image_size * data.filter_w / data.slot_count + l] = ct;
        }

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_ct_per_V; ct_ind++) {
            vector<Ciphertext> rotation_results(num_diag);

            #pragma omp parallel for
            for (int i = 0; i < num_diag; i++) {
                Ciphertext temp1;
                Ciphertext temp2;
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(he, data, V[packing_ind * num_ct_per_V + ct_ind], i);
                he->evaluator->multiply_plain(temp_mult[0], softmax_s2[packing_ind][0][i], temp1);
                he->evaluator->multiply_plain(temp_mult[1], softmax_s2[packing_ind][1][i], temp2);
                he->evaluator->add(temp1, temp2, rotation_results[i]);
            }
            for (int i = 0; i < num_diag; i++) {
                he->evaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w / data.slot_count + ct_ind], rotation_results[i]);
            }
            rotation_results.clear();
        }
    }
    for (int i = 0; i < result.size(); i++) {
        he->evaluator->mod_switch_to_next_inplace(result[i]);
    }
}
