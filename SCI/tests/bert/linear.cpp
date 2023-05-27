#include "linear.h"

Linear::Linear(){}

Linear::Linear(int party, NetIO *io) {
	this->party = party;
	this->io = io;
	this->slot_count = 8192;

	this->party = party;
	this->io = io;
	this->slot_count = POLY_MOD_DEGREE;
}

Linear::~Linear() {

}

// // Generate the masks for 1-step rotation
// vector<vector<Plaintext>> Linear::generate_rotation_masks() {
//     vector<vector<Plaintext>> result;
//     for (int i = 0; i < 128; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         vector<int64_t> mask2(data.slot_count, 0LL);
//         for (int j = 0; j < 128 - i; j++) {
//             for (int k = 0; k < 32; k++) {
//                 mask1[j + 128 * k] = 1;
//                 mask1[j + 128 * k + data.slot_count / 2] = 1;
//             }
//         }
//         for (int j = 128 - i; j < 128; j++) {
//             for (int k = 0; k < 32; k++) {
//                 mask2[j + 128 * k] = 1;
//                 mask2[j + 128 * k + data.slot_count / 2] = 1;
//             }
//         }
//         Plaintext pt1;
//         Plaintext pt2;
//         encoder->encode(mask1, pt1);
//         encoder->encode(mask2, pt2);
//         result.push_back({pt1, pt2});
//     }
//     return result;
// }

// // Generate cipher_masks: 1111100000..., 0000011111..., ...
// vector<Plaintext> Linear::generate_cipher_masks() {
//     vector<Plaintext> result;
//     for (int i = 0; i < 32; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         for (int k = 0; k < 128; k++)
//             mask1[i * 128 + k] = 1;
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }

//     for (int i = 0; i < 32; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         for (int k = 0; k < 128; k++)
//             mask1[i * 128 + k + data.slot_count / 2] = 1;
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }

//     for (int i = 0; i < 32; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         for (int k = 0; k < 128; k++)
//             mask1[i * 128 + k] = 1;
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }

//     for (int i = 0; i < 32; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         for (int k = 0; k < 128; k++)
//             mask1[i * 128 + k + data.slot_count / 2] = 1;
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }
//     return result;
// }

// vector<Plaintext> Linear::generate_depth3_masks() {
//     vector<Plaintext> result;

//     for (int i = 0; i < 64; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         for (int k = 0; k < 128 - i; k++)
//             mask1[i * 128 + k] = 1;
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }

//     for (int i = 0; i < 64; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         for (int k = 0; k < 128 - i - 64; k++)
//             mask1[i * 128 + k] = 1;
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }

//     for (int i = 0; i < 64; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         for (int k = 128 - i; k < 128; k++)
//             mask1[i * 128 + k] = 1;
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }

//     for (int i = 0; i < 64; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         for (int k = 128 - i - 64; k < 128; k++)
//             mask1[i * 128 + k] = 1;
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }
//     return result;
// }

// vector<Plaintext> Linear::generate_cross_packing_masks() {
//     vector<Plaintext> result;

//     for (int i = 0; i < 32; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         if (i == 0) {
//             for (int k = 0; k < 128 - i; k++)
//                 mask1[k] = 1;
//         }
//         else {
//             for (int k = 0; k < 128 - i; k++)
//                 mask1[i * 128 + k] = 1;
//             for (int k = 0; k < 128 - i; k++)
//                 mask1[i * 128 + k + data.slot_count / 2] = 1;
//         }
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }

//     for (int i = 32; i <= 64; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         if (i == 64) {
//             for (int k = 0; k < 128 - i; k++)
//                 mask1[k + data.slot_count / 2] = 1;
//         }
//         else {
//             for (int k = 0; k < 128 - i; k++)
//                 mask1[(i - 32) * 128 + k] = 1;
//             for (int k = 0; k < 128 - i; k++)
//                 mask1[(i - 32) * 128 + k + data.slot_count / 2] = 1;
//         }
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }

//     for (int i = 0; i < 32; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         if (i == 0) {
//             for (int k = 128 - i; k < 128; k++)
//                 mask1[k] = 1;
//         }
//         else {
//             for (int k = 128 - i; k < 128; k++)
//                 mask1[i * 128 + k] = 1;
//             for (int k = 128 - i; k < 128; k++)
//                 mask1[i * 128 + k + data.slot_count / 2] = 1;
//         }
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }
//     for (int i = 32; i <= 64; i++) {
//         vector<int64_t> mask1(data.slot_count, 0LL);
//         if (i == 64) {
//             for (int k = 128 - i; k < 128; k++)
//                 mask1[k + data.slot_count / 2] = 1;
//         }
//         else {
//             for (int k = 128 - i; k < 128; k++)
//                 mask1[(i - 32) * 128 + k] = 1;
//             for (int k = 128 - i; k < 128; k++)
//                 mask1[(i - 32) * 128 + k + data.slot_count / 2] = 1;
//         }
//         Plaintext pt;
//         encoder->encode(mask1, pt);
//         result.push_back(pt);
//     }
//     return result;
// }

// // column-wise packing
// vector<Ciphertext> Linear::bert_efficient_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data) {

//     vector<int64_t> pod_matrix(data.slot_count, 0ULL);
//     vector<Ciphertext> cts;
//     for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
//     {
//         pod_matrix = vector<int64_t>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
//         Ciphertext ct;
//         Plaintext pt;
//         encoder->encode(pod_matrix, pt);
//         encryptor->encrypt(pt, ct);
//         cts.push_back(ct);
//     }
//     return cts;
// }

// void Linear::bert_cipher_cipher_cross_packing(const FCMetadata &data, const vector<Ciphertext> &Cipher_plain_result, const vector<Plaintext> &cross_masks, vector<Ciphertext> &results)
// {
//     int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;

//     for (int packing_index = 0; packing_index < 12; packing_index++) {
//         Ciphertext HE_result_1_left = Cipher_plain_result[0 + packing_gap * packing_index];
//         Ciphertext HE_result_2_left = Cipher_plain_result[1 + packing_gap * packing_index];

//         Ciphertext HE_result_1_right;
//         Ciphertext HE_result_2_right;

//         evaluator->rotate_columns(HE_result_1_left, *gal_keys, HE_result_1_right);
//         evaluator->rotate_columns(HE_result_2_left, *gal_keys, HE_result_2_right);

//         vector<Ciphertext> rotation_results(data.image_size + 2);
//         auto t1 = high_resolution_clock::now();
//         vector<Ciphertext> rotation_results_left(data.image_size + 2);
//         vector<Ciphertext> rotation_results_right(data.image_size + 2);

//         #pragma omp parallel for
//         for (int i = 0; i <= data.image_size / 2; i++) {
//             vector<Ciphertext> temp_mult = rotation_by_one_depth3(data, HE_result_1_right, i);

//             evaluator->multiply(HE_result_1_left, temp_mult[0], rotation_results_left[i]);
//             evaluator->relinearize_inplace(rotation_results_left[i], *relin_keys);

//             evaluator->multiply(HE_result_1_left, temp_mult[1], rotation_results_left[i + data.image_size / 2 + 1]);
//             evaluator->relinearize_inplace(rotation_results_left[i + data.image_size / 2 + 1], *relin_keys);

//             temp_mult = rotation_by_one_depth3(data, HE_result_2_right, i);

//             evaluator->multiply(HE_result_2_left, temp_mult[0], rotation_results_right[i]);
//             evaluator->relinearize_inplace(rotation_results_right[i], *relin_keys);

//             evaluator->multiply(HE_result_2_left, temp_mult[1], rotation_results_right[i + data.image_size / 2 + 1]);
//             evaluator->relinearize_inplace(rotation_results_right[i + data.image_size / 2 + 1], *relin_keys);

//             evaluator->add(rotation_results_left[i], rotation_results_right[i], rotation_results[i]);
//             evaluator->add(rotation_results_left[i + data.image_size / 2 + 1], rotation_results_right[i + data.image_size / 2 + 1], rotation_results[i + data.image_size / 2 + 1]);
//         }
//         auto t2 = high_resolution_clock::now();
//         auto ms_double = (t2 - t1)/1e+9;
//         std::cout << "[Server] Cipher-Cipher Rotation 1 " << ms_double.count() << std::endl;

//         t1 = high_resolution_clock::now();
//         int local_rotation = std::ceil(std::log2(32));
//         #pragma omp parallel for
//         for (int i = 0; i < data.image_size + 2; i++) {
//             for (int k = 0; k < local_rotation; k++) {
//                 Ciphertext temp2;
//                 evaluator->rotate_rows(rotation_results[i], (int32_t) pow(2, k) * 128, *gal_keys, temp2);
//                 evaluator->add_inplace(rotation_results[i], temp2);
//             }
//             evaluator->multiply_plain_inplace(rotation_results[i], cross_masks[i]);
//         }
//         t2 = high_resolution_clock::now();
//         ms_double = (t2 - t1)/1e+9;
//         std::cout << "[Server] Cipher-Cipher Rotation 2 " << ms_double.count() << std::endl;
//         // Packing
//         t1 = high_resolution_clock::now();
        
//         evaluator->add(rotation_results[0], rotation_results[65], results[0 + 2 * packing_index]);
//         evaluator->add(rotation_results[32], rotation_results[32 + 65], results[1 + 2 * packing_index]);

//         for (int i = 1; i < 32; i++) {
//             Ciphertext temp;
//             evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[i]);
//             evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[i + 65]);
//             evaluator->add_inplace(results[1 + 2 * packing_index], rotation_results[i + 32]);
//             evaluator->add_inplace(results[1 + 2 * packing_index], rotation_results[i + 32 + 65]);
//         }

//         evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[64]);
//         evaluator->add_inplace(results[0 + 2 * packing_index], rotation_results[64 + 65]);
//         t2 = high_resolution_clock::now();
//         ms_double = (t2 - t1)/1e+9;
//         std::cout << "[Server] Cipher-Cipher Packing " << ms_double.count() << std::endl;
//     }
// }

// uint64_t* Linear::bert_cross_packing_postprocess(vector<Ciphertext> &cts, const FCMetadata &data) {
//     uint64_t *result = new uint64_t[data.image_size*data.image_size];
//     for (int i = 0; i < cts.size(); i++) {
//         vector<int64_t> plain(data.slot_count, 0ULL);
//         Plaintext tmp;
//         decryptor->decrypt(cts[i], tmp);
//         encoder->decode(tmp, plain);

//         #pragma omp parallel for
//         for (int row = 0; row < data.slot_count; row++) {
//             int j = row / data.image_size;
//             int k = row % data.image_size;
//             if (j < 32) { // k, (k + j) % 128
//                 result[k + ((k + j + i * 32) % data.image_size) * data.image_size] = plain[row];
//             }
//             else if (j == 32 && i == 0) { // (64 + k) % 128, k
//                 result[((k + 64) % data.image_size) + k * data.image_size] = plain[row];
//             }
//             else { // (k - 32 + j) % 128, k
//                 result[k * data.image_size + (k + j - 32 + i * 32) % 128] = plain[row];
//             }
//         }
//     }
//     return result;
// }

// vector<Ciphertext> Linear::rotation_by_one_depth3(const FCMetadata &data, const Ciphertext &ct, int k) {

//     int m = -(128 - k);
//     Ciphertext ct1;
//     Ciphertext ct2;
//     evaluator->rotate_rows(ct, k, *gal_keys, ct1);
//     evaluator->rotate_rows(ct, m, *gal_keys, ct2);

//     return {ct1, ct2};
// }

// void Linear::matrix_multiplication(
// 	int32_t input_dim, 
//     int32_t common_dim, 
//     int32_t output_dim, 
//     vector<vector<uint64_t>> &A, 
//     vector<vector<uint64_t>> &B1, 
// 	vector<vector<uint64_t>> &B2, 
//     vector<vector<uint64_t>> &C, 
//     bool verify_output) {

//     data.filter_h = common_dim;
//     data.filter_w = output_dim;
//     data.image_size = input_dim;
//     this->slot_count = 8192;
//     configure();
    
//     if (party == BOB) {  
//         // Client
//         vector<uint64_t> vec(common_dim * input_dim);
//         for (int j = 0; j < common_dim; j++)
//             for (int i = 0; i < input_dim; i++)
//                 vec[j*input_dim + i] = A[i][j];

//         auto cts = bert_efficient_preprocess_vec(vec, data);
//         auto io_start = io->counter;
//         send_encrypted_vector(io, cts);
//         cout << "[Client] Input cts sent" << endl;
//         cout << "[Client] Size of cts (Bytes): " << sizeof(Ciphertext) << " " << sizeof(Ciphertext) * cts.size() << endl;

//         vector<Ciphertext> enc_result(2 * 12);
//         recv_encrypted_vector(context, io, enc_result);
//         cout << "[Client] Output cts received" << endl;
//         cout << "[Client] size of cts (Bytes): " << io->counter - io_start << endl;

//         print_noise_budget_vec(enc_result);
//         // print_ct(enc_result[0], data.slot_count);

//         // auto HE_result = bert_efficient_postprocess(enc_result, data);
//         auto HE_result = bert_cross_packing_postprocess(enc_result, data);

//         #ifdef HE_DEBUG
//         for (int i = 64; i < 67; i++) {
//             for (int j = 0; j < 128; j++)
//                 cout << ((int64_t) HE_result[i + j * 128] + (int64_t) prime_mod) % (int64_t) prime_mod << " ";
//             cout << endl;
//         }
//         #endif


//         delete[] HE_result;
//     } else {
//         // Server
//         #ifdef HE_TIMING
//         auto t1_total = high_resolution_clock::now();
//         #endif 

//         auto io_start = io->counter;
//         vector<Ciphertext> cts(12);
//         recv_encrypted_vector(this->context, io, cts);

//         #ifdef HE_TIMING
//         auto t1_preprocess = high_resolution_clock::now();
//         #endif

//         vector<uint64_t *> matrix_mod_p1(common_dim);
//         vector<uint64_t *> matrix_mod_p2(common_dim);
//         vector<uint64_t *> matrix_mod_p3(common_dim);
//         vector<uint64_t *> matrix_mod_p4(common_dim);
//         vector<uint64_t *> matrix1(common_dim);
//         vector<uint64_t *> matrix2(common_dim);
//         for (int i = 0; i < common_dim; i++) {
//             matrix_mod_p1[i] = new uint64_t[output_dim];
//             matrix_mod_p2[i] = new uint64_t[output_dim];
//             matrix_mod_p3[i] = new uint64_t[output_dim / 2];
//             matrix_mod_p4[i] = new uint64_t[output_dim / 2];
//             matrix1[i] = new uint64_t[output_dim];
//             matrix2[i] = new uint64_t[output_dim];
//             for (int j = 0; j < output_dim; j++) {
//                 matrix_mod_p1[i][j] = neg_mod((int64_t)B1[i][j], (int64_t)prime_mod);
//                 matrix_mod_p2[i][j] = neg_mod((int64_t)B2[i][j], (int64_t)prime_mod);
//                 int64_t val = (int64_t)B1[i][j];
//                 if (val > int64_t(prime_mod / 2)) {
//                     val = val - prime_mod;
//                 }
//                 matrix1[i][j] = val;
//                 val = (int64_t)B2[i][j];
//                 if (val > int64_t(prime_mod / 2)) {
//                     val = val - prime_mod;
//                 }
//                 matrix2[i][j] = val;
//             }

//             for (int j = 0; j < output_dim / 2; j++) {
//                 matrix_mod_p3[i][j] = neg_mod((int64_t)B1[i][j], (int64_t)prime_mod);
//                 matrix_mod_p4[i][j] = neg_mod((int64_t)B1[i][j + output_dim / 2], (int64_t)prime_mod);
//             }
//         }

//         PRG128 prg;
//         uint64_t *secret_share = new uint64_t[input_dim*output_dim];
//         prg.random_mod_p<uint64_t>(secret_share, input_dim*output_dim, prime_mod);
//         // auto encoded_mat1 = bert_efficient_preprocess_matrix(matrix_mod_p1.data(), data);
//         // auto encoded_mat2 = bert_efficient_preprocess_matrix(matrix_mod_p2.data(), data);

//         // auto cross_mat = bert_cross_packing_matrix(matrix_mod_p1.data(), matrix_mod_p2.data(), data);
//         // auto cross_mat = bert_cross_packing_matrix_bsgs(matrix_mod_p1.data(), matrix_mod_p2.data(), data);
//         vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats(12);
//         vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> cross_mats_single(12);

//         for (int i = 0; i < 12; i++) {
//             auto cross_mat = bert_cross_packing_matrix(matrix_mod_p1.data(), matrix_mod_p2.data(), data);
//             auto cross_mat_single = bert_cross_packing_single_matrix(matrix_mod_p3.data(), matrix_mod_p4.data(), data);
//             cross_mats[i] = cross_mat;
//             cross_mats_single[i] = cross_mat_single;
//         }

//         auto rotation_masks = generate_rotation_masks();
//         auto cipher_masks = generate_cipher_masks();
//         auto depth3_masks = generate_depth3_masks();
//         auto cross_masks = generate_cross_packing_masks();

//         // Ciphertext enc_noise = bert_efficient_preprocess_noise(secret_share, data, cryptoContext_, keyPair_);
//         // cout << "[Server] Noise processed" << endl;
//         #ifdef HE_TIMING
//         auto t2_preprocess = high_resolution_clock::now();
//         auto interval = (t2_preprocess - t1_preprocess)/1e+9;
//         cout << "[Server] Preprocessing takes " << interval.count() << "sec" << endl;
//         #endif

//         #ifdef HE_DEBUG
//             print_noise_budget_vec(cts);
//         #endif

//         #ifdef HE_TIMING
//         auto t1_cipher_plain = high_resolution_clock::now();
//         #endif 

//         vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w / data.slot_count * 3 * 12);
//         bert_cipher_plain_bsgs(cts, cross_mats, cross_mats_single, data, Cipher_plain_results);

//         #ifdef HE_TIMING
//         auto t2_cipher_plain = high_resolution_clock::now();
//         interval = (t2_cipher_plain - t1_cipher_plain)/1e+9;
//         cout << "[Server] Cipher-Plaintext Matmul takes " << interval.count() << "sec" << endl;

//         auto t1_cipher_cipher = high_resolution_clock::now();
//         #endif 

//         // auto HE_result = bert_efficient_cipher(data, Cipher_plain_results, rotation_masks, cipher_masks);
//         // auto HE_result = bert_efficient_cipher_depth3(data, Cipher_plain_results, depth3_masks);
//         vector<Ciphertext> HE_result(2 * 12);
//         bert_cipher_cipher_cross_packing(data, Cipher_plain_results, cross_masks, HE_result);

//         #ifdef HE_TIMING
//         auto t2_cipher_cipher = high_resolution_clock::now();
//         interval = (t2_cipher_cipher - t1_cipher_cipher)/1e+9;
//         cout << "[Server] Cipher-Cipher Matmul takes " << interval.count() << "sec" << endl;
//         #endif 

//         send_encrypted_vector(io, HE_result);

//         cout << "[Server] Result sent" << endl;
//         cout << "[Server] size of result (Bytes): " << io->counter - io_start << endl;

//         for (int i = 0; i < common_dim; i++) {
//             delete[] matrix_mod_p1[i];
//             delete[] matrix_mod_p2[i];
//             delete[] matrix1[i];
//             delete[] matrix2[i];
//         }
//         delete[] secret_share;

//         #ifdef HE_TIMING
//         auto t2_total = high_resolution_clock::now();
//         interval = (t2_total - t1_total)/1e+9;
//         cout << "[Server] Total Time " << interval.count() << "sec" << endl;
//         #endif 
//     }
// }