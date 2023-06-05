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
	this->he_37 = new HE(
		party,
		io,
		4096,
		{40, 39, 30},
		(uint64_t) pow(2, 37)
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

PreprocessParams_1 Linear::params_preprocessing_ct_pt_1(
    HE* he,
    vector<vector<vector<uint64_t>>> w_q,
    vector<vector<vector<uint64_t>>> w_k,
    vector<vector<vector<uint64_t>>> w_v,
    vector<vector<uint64_t>> b_q,
    vector<vector<uint64_t>> b_k,
    vector<vector<uint64_t>> b_v,
    const FCMetadata &data
){
    int input_dim = 128;
    int common_dim = 768;
    int output_dim = 64;

    uint64_t plain_mod = he->plain_mod;

    PreprocessParams_1 pp;
     for (int packing_index = 0; packing_index < 12; packing_index++) {
        vector<uint64_t *> matrix_mod_p1(common_dim);
        vector<uint64_t *> matrix_mod_p2(common_dim);
        vector<uint64_t *> matrix_mod_p3(common_dim);

        vector<uint64_t> bias_mod_p1(output_dim);
        vector<uint64_t> bias_mod_p2(output_dim);
        vector<uint64_t> bias_mod_p3(output_dim);
        for (int i = 0; i < common_dim; i++) {
            matrix_mod_p1[i] = new uint64_t[output_dim];
            matrix_mod_p2[i] = new uint64_t[output_dim];
            matrix_mod_p3[i] = new uint64_t[output_dim];

            for (int j = 0; j < output_dim; j++) {
                matrix_mod_p1[i][j] = neg_mod((int64_t)w_q[packing_index][i][j], (int64_t)plain_mod);
                matrix_mod_p2[i][j] = neg_mod((int64_t)w_k[packing_index][i][j], (int64_t)plain_mod);
                matrix_mod_p3[i][j] = neg_mod((int64_t)w_v[packing_index][i][j], (int64_t)plain_mod);
            }
        }

        for (int i = 0; i < output_dim; i++) {
            bias_mod_p1[i] = neg_mod((int64_t)b_q[packing_index][i], (int64_t)plain_mod);
            bias_mod_p2[i] = neg_mod((int64_t)b_k[packing_index][i], (int64_t)plain_mod);
            bias_mod_p3[i] = neg_mod((int64_t)b_v[packing_index][i], (int64_t)plain_mod);
        }

        auto encoded_mat1 = preprocess_matrix(matrix_mod_p1.data(), data);
        auto encoded_mat2 = preprocess_matrix(matrix_mod_p2.data(), data);
        auto encoded_mat3 = preprocess_matrix(matrix_mod_p2.data(), data);

        auto temp_bias1 = preprocess_bias(bias_mod_p1.data(), data);
        auto temp_bias2 = preprocess_bias(bias_mod_p2.data(), data);
        auto temp_bias3 = preprocess_bias(bias_mod_p3.data(), data);
        pp.encoded_mats1.push_back(encoded_mat1);
        pp.encoded_mats2.push_back(encoded_mat2);
        pp.encoded_mats3.push_back(encoded_mat3);
        pp.encoded_bias1.push_back(temp_bias1);
        pp.encoded_bias2.push_back(temp_bias2);
        pp.encoded_bias3.push_back(temp_bias3);
    }

    return pp;
}

PreprocessParams_2 Linear::params_preprocessing_ct_pt_2(
    HE* he,
    int32_t input_dim, 
    int32_t common_dim, 
    int32_t output_dim,
    vector<vector<uint64_t>> w,
    vector<uint64_t> b,
    const FCMetadata &data
){
    uint64_t plain_mod = he->plain_mod;
    PreprocessParams_2 pp;

    vector<uint64_t *> matrix_mod_p1(common_dim);
    vector<uint64_t *> matrix1(common_dim);
    for (int i = 0; i < common_dim; i++) {
        matrix_mod_p1[i] = new uint64_t[output_dim];
        matrix1[i] = new uint64_t[output_dim];
        for (int j = 0; j < output_dim; j++) {
            matrix_mod_p1[i][j] = neg_mod((int64_t)w[i][j], (int64_t)plain_mod);
        }
    }
    for (int i = 0; i < output_dim; i++) {
        b[i] = neg_mod((int64_t)b[i], (int64_t)plain_mod);
    }

    // PRG128 prg;
    // uint64_t *secret_share = new uint64_t[input_dim*output_dim];
    // prg.random_mod_p<uint64_t>(secret_share, input_dim*output_dim, prime_mod);

    pp.encoded_mat = preprocess_matrix(matrix_mod_p1.data(), data);
    pp.encoded_bias = preprocess_bias(b.data(), data);
}

vector<vector<Plaintext>> Linear::preprocess_matrix(const uint64_t *const *matrix, const FCMetadata &data){
    vector<vector<Plaintext>> weightMatrix;
    int sub_mat_row = data.filter_h / data.nw; // 48
    int sub_mat_col = data.filter_w / data.kw; // 32
    for (int sub_row_ind = 0; sub_row_ind < sub_mat_row; sub_row_ind++) {
        vector<Plaintext> temp;
        for (int sub_col_ind = 0; sub_col_ind < sub_mat_col; sub_col_ind++) {
            vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
            for (int i = 0; i < data.nw; i++) {
                for (int j = 0; j < data.kw; j++) {
                    pod_matrix[j * data.nw + i] = matrix[i + sub_row_ind * data.nw][j + sub_col_ind * data.kw];
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

vector<Plaintext> Linear::preprocess_bias(const uint64_t *matrix, const FCMetadata &data){
    int res_cts_num = data.filter_w / data.kw;
    vector<Plaintext> packed_bias(res_cts_num);
    for (int ct_ind = 0; ct_ind < res_cts_num; ct_ind++) {
        vector<uint64_t> pt_data(data.slot_count, 0ULL);
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < data.kw; j++) {
                pt_data[i * data.nw * data.kw + (j + 1) * data.nw - 1] = matrix[(j + ct_ind * data.kw)];
            }
        }
        packed_bias[ct_ind] = encode_vector(pt_data.data(), data);
    }
    return packed_bias;
}

Plaintext Linear::encode_vector(const uint64_t *vec, const FCMetadata &data) {
    Plaintext pt;
    pt.resize(data.slot_count);
    assert(pt.data() != nullptr);
    seal::util::modulo_poly_coeffs(vec, data.slot_count, prime_mod, pt.data());
    return pt;
}