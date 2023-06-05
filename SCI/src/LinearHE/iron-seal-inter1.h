#ifndef BERTFC_FIELD_H__
#define BERTFC_FIELD_H__

#include "utils-HE.h"
#include <fstream>

using namespace std;
using namespace sci;
using namespace seal;

struct FCMetadata {
    int slot_count;
    int32_t pack_num;
    int32_t inp_ct;
    // Filter is a matrix
    int32_t filter_h;
    int32_t filter_w;
    int32_t filter_size;
    // Image is a matrix
    int32_t image_size;
    int32_t nw;
    int32_t kw;
};

class IRONINT1 {
public:
    int party;
    NetIO *io;
    FCMetadata data;
    SEALContext *context;
    Encryptor *encryptor;
    Decryptor *decryptor;
    Evaluator *evaluator;
    BatchEncoder *encoder;
    GaloisKeys *gal_keys;
    RelinKeys *relin_keys;
    Ciphertext *zero;
    size_t slot_count;

    IRONINT1(int party, NetIO *io);

    ~IRONINT1();

    void configure();

    Plaintext encode_vector(const uint64_t *vec, const FCMetadata &data);

    vector<Ciphertext> preprocess_vec(vector<uint64_t> &input, const FCMetadata &data);

    vector<vector<Plaintext>> preprocess_matrix(const uint64_t *const *matrix, const FCMetadata &data);

    vector<Plaintext> preprocess_bias(const uint64_t *matrix, const FCMetadata &data);

    Ciphertext preprocess_noise(const uint64_t *secret_share, const FCMetadata &data);

    vector<Ciphertext> bert_cipher_plain(vector<Ciphertext> &cts, vector<vector<Plaintext>> &encoded_mat1, vector<Plaintext> &encoded_bias, const FCMetadata &data);

    uint64_t* bert_postprocess(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing);

    vector<uint64_t> ideal_functionality(uint64_t *vec, uint64_t **matrix);

    void print_noise_budget_vec(vector<Ciphertext> v);

    void print_ct(Ciphertext &ct, int len);
    void print_pt(Plaintext &pt, int len);
    void saveMatrix(const std::string& filename, uint64_t* matrix, size_t rows, size_t cols);

    void matrix_multiplication(int32_t input_dim, int32_t common_dim,
                                int32_t output_dim,
                                vector<vector<uint64_t>> &A,
                                vector<vector<uint64_t>> &B,
                                vector<uint64_t> &Bias,
                                vector<vector<uint64_t>> &C,
                                bool verify_output = false);
};
#endif
