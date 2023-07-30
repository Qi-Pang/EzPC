#include "LinearHE/bert-layernorm.h"
#include <omp.h>

using namespace std;
using namespace sci;
using namespace seal;

#define HE_TIMING
// #define HE_DEBUG

void LayerNormField::saveMatrix(const std::string& filename, uint64_t* matrix, size_t rows, size_t cols) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                // file << matrix[i * cols + j];
                file << ((int64_t) matrix[i * cols + j] + (int64_t) prime_mod) % (int64_t) prime_mod;
                if (j < cols - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
    } 
    else {
        std::cout << "Unable to open file";
    }
}

void LayerNormField::print_noise_budget_vec(vector<Ciphertext> v) {
    cout << "Noise budget: ";
    for(int i = 0; i < v.size(); i++){
        cout << YELLOW << decryptor->invariant_noise_budget(v[i]) << " ";
    }
    cout << RESET << endl;
}

void LayerNormField::print_ct(Ciphertext &ct, int len){
    Plaintext pt;
    decryptor->decrypt(ct, pt);
    print_pt(pt, len);
}

void LayerNormField::print_pt(Plaintext &pt, int len) {
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


LayerNormField::LayerNormField(int party, NetIO *io) {
    this->party = party;
    this->io = io;
    this->slot_count = 8192;

    this->party = party;
    this->io = io;
    generate_new_keys_layernorm(party, io, slot_count, context, encryptor, decryptor,
                    evaluator, encoder, zero);
}

LayerNormField::~LayerNormField() {
    free_keys_layernorm(party, encryptor, decryptor, evaluator, encoder, zero);
}

void LayerNormField::configure() {
  data.slot_count = 8192;
  // Only works with a ciphertext that fits in a single ciphertext
  assert(data.slot_count >= data.image_size);

  data.filter_size = data.filter_h * data.filter_w;
  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / next_pow2(data.filter_w);
  // How many total ciphertexts we'll need
  data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
}

// Below computes the gamma * x
// Gamma (768) * (x1 + x2) (128 x 768) * var (128)
vector<Plaintext> LayerNormField::pack_gamma(vector<uint64_t> &gamma, const FCMetadata &data) {
    int col_per_ct = data.slot_count / data.image_size;
    int cts_size = data.image_size * data.filter_h / data.slot_count;
    vector<Plaintext> result(cts_size);
    for (int i = 0; i < cts_size; i++) {
        vector<uint64_t> temp_pack(data.slot_count, 0ULL);
        for (int k = 0; k < col_per_ct; k++) {
            for (int j = 0; j < data.image_size; j++) {
                temp_pack[j + k * data.image_size] = gamma[i * col_per_ct + k];
            }
        }
        Plaintext pt1;
        encoder->encode(temp_pack, pt1);
        result[i] = pt1;
    }
    return result;
}

vector<Plaintext> LayerNormField::pack_x_plain(vector<vector<uint64_t>> &x, const FCMetadata &data) {
    int cts_size = data.image_size * data.filter_h / data.slot_count;
    vector<Plaintext> result(cts_size);
    for (int i = 0; i < cts_size; i++) {
        vector<uint64_t> temp_pack(data.slot_count, 0ULL);
        for (int ind = 0; ind < data.slot_count; ind++) {
            int row = ind % data.image_size;
            int col = ind / data.image_size + i * data.slot_count / data.image_size;
            temp_pack[ind] = neg_mod((int64_t)x[row][col], (int64_t)prime_mod);
        }
        Plaintext pt1;
        encoder->encode(temp_pack, pt1);
        result[i] = pt1;
    }
    return result;
}

vector<Ciphertext> LayerNormField::pack_x_cipher(vector<vector<uint64_t>> &x, const FCMetadata &data) {
    int cts_size = data.image_size * data.filter_h / data.slot_count;
    vector<Ciphertext> result(cts_size);
    for (int i = 0; i < cts_size; i++) {
        vector<uint64_t> temp_pack(data.slot_count, 0ULL);
        for (int ind = 0; ind < data.slot_count; ind++) {
            int row = ind % data.image_size;
            int col = ind / data.image_size + i * data.slot_count / data.image_size;
            temp_pack[ind] = neg_mod((int64_t)x[row][col], (int64_t)prime_mod);
        }
        Plaintext pt1;
        Ciphertext ct1;
        encoder->encode(temp_pack, pt1);
        encryptor->encrypt(pt1, ct1);
        result[i] = ct1;
    }
    return result;
}

vector<Plaintext> LayerNormField::gamma_x2_plain_server(vector<vector<uint64_t>> &x2_pt, vector<uint64_t> &gamma, vector<vector<uint64_t>> &sharing_r, const FCMetadata &data) {
    vector<vector<uint64_t>> result(data.image_size, vector<uint64_t>(data.filter_h));
    for (int i = 0; i < data.image_size; i++) {
        for (int j = 0; j < data.filter_h; j++) {
            result[i][j] = neg_mod((int64_t)(x2_pt[i][j] * gamma[j] - sharing_r[i][j]), (int64_t)prime_mod);
        }
    }
    return pack_x_plain(result, data);
}

vector<Ciphertext> LayerNormField::gamma_x_server(vector<Ciphertext> &x1_ct, vector<Plaintext> &enc_gamma, vector<Plaintext> &gamma_x2_pt, const FCMetadata &data) {
    int cts_size = x1_ct.size();
    vector<Ciphertext> result(cts_size);
    for (int i = 0; i < cts_size; i++) {
        evaluator->multiply_plain(x1_ct[i], enc_gamma[i], result[i]);
        evaluator->add_plain_inplace(result[i], gamma_x2_pt[i]);
    }
    return result;
}

// Below computes (128 x 768) y * var (768)
vector<vector<uint64_t>> LayerNormField::y_var_plain(vector<vector<uint64_t>> &y1, vector<uint64_t> &var1, const FCMetadata &data) {
    for (int i = 0; i < data.image_size; i++) {
        for (int j = 0; j < data.filter_h; j++) {
            y1[i][j] *= var1[j];
        }
    }
    return y1;
}

vector<Plaintext> LayerNormField::pack_var_plain(vector<uint64_t> &var, const FCMetadata &data) {
    int cts_size = data.image_size * data.filter_h / data.slot_count;
    vector<Plaintext> result(cts_size);
    for (int i = 0; i < cts_size; i++) {
        vector<uint64_t> temp_pack(data.slot_count, 0ULL);
        for (int ind = 0; ind < data.slot_count; ind++) {
            int col = ind / data.image_size + i * data.slot_count / data.image_size;
            temp_pack[ind] = neg_mod((int64_t)var[col], (int64_t)prime_mod);
        }
        Plaintext pt1;
        encoder->encode(temp_pack, pt1);
        result[i] = pt1;
    }
    return result;
}

vector<Ciphertext> LayerNormField::pack_var_cipher(vector<uint64_t> &var, const FCMetadata &data) {
    int cts_size = data.image_size * data.filter_h / data.slot_count;
    vector<Ciphertext> result(cts_size);
    for (int i = 0; i < cts_size; i++) {
        vector<uint64_t> temp_pack(data.slot_count, 0ULL);
        for (int ind = 0; ind < data.slot_count; ind++) {
            int col = ind / data.image_size + i * data.slot_count / data.image_size;
            temp_pack[ind] = var[col];
        }
        Plaintext pt1;
        Ciphertext ct1;
        encoder->encode(temp_pack, pt1);
        encryptor->encrypt(pt1, ct1);
        result[i] = ct1;
    }
    return result;
}

vector<Ciphertext> LayerNormField::y1_var2_server(vector<Ciphertext> &y1_ct, vector<Plaintext> &var2_pt, const FCMetadata &data) {
    int cts_size = data.image_size * data.filter_h / data.slot_count;
    for (int i = 0; i < cts_size; i++) {
        evaluator->multiply_plain_inplace(y1_ct[i], var2_pt[i]);
    }
    return y1_ct;
}

vector<Ciphertext> LayerNormField::var1_y2_server(vector<Ciphertext> &var1_ct, vector<Plaintext> &y2_pt, const FCMetadata &data) {
    int cts_size = data.image_size * data.filter_h / data.slot_count;
    for (int i = 0; i < cts_size; i++) {
        evaluator->multiply_plain_inplace(var1_ct[i], y2_pt[i]);
    }
    return var1_ct;
}

// Below computes the (x - u) * (x - u)

vector<vector<uint64_t>> LayerNormField::x_square_plain(vector<vector<uint64_t>> &x, const FCMetadata &data) {
    for (int i = 0; i < data.image_size; i++) {
        for (int j = 0; i < data.filter_h; j++) {
            x[i][j] *= x[i][j];
        }
    }
    return x;
}

vector<Ciphertext> LayerNormField::x1_x2_server(vector<Ciphertext> &x1, vector<Plaintext> &x2, const FCMetadata &data) {
    int cts_size = data.image_size * data.filter_h / data.slot_count;
    for (int i = 0; i < cts_size; i++) {
        evaluator->multiply_plain_inplace(x1[i], x2[i]);
    }
    return x1;
}

vector<uint64_t> LayerNormField::postprocess(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing) {
    vector<uint64_t> result(data.image_size * data.filter_h);
    for (int ct_ind = 0; ct_ind < cts.size(); ct_ind++) {
        vector<uint64_t> temp_plain(data.slot_count, 0ULL);
        Plaintext pt;
        decryptor->decrypt(cts[ct_ind], pt);
        encoder->decode(pt, temp_plain);
        for (int pt_ind = 0; pt_ind < data.slot_count; pt_ind++) {
            int row = pt_ind % data.image_size;
            int col = pt_ind / data.image_size + ct_ind * data.slot_count / data.image_size;
            if (col_packing) {
                result[row + col * data.image_size] = temp_plain[pt_ind];
            }
            else {
                result[col + row * data.filter_h] = temp_plain[pt_ind];
            }
        }
    }
    return result;
}

void LayerNormField::layernorm_he(int32_t input_dim, 
                                    int32_t common_dim, 
                                    int32_t output_dim, 
                                    vector<vector<uint64_t>> &X1, 
                                    vector<vector<uint64_t>> &X2, 
                                    vector<uint64_t> &Gamma, 
                                    vector<uint64_t> &Var1, 
                                    vector<uint64_t> &Var2) {

    data.filter_h = common_dim;
    data.filter_w = output_dim;
    data.image_size = input_dim;
    this->slot_count = 8192;
    configure();

    if (party == BOB) {
        // Client

        vector<Ciphertext> x1_ct = pack_x_cipher(X1, data);
        // vector<Ciphertext> var1_ct = pack_var_cipher(Var1, data);
        // x1_ct.insert(x1_ct.end(), var1_ct.begin(), var1_ct.end());
        send_encrypted_vector(io, x1_ct);
        int cts_size = data.image_size * data.filter_h / data.slot_count;
        vector<Ciphertext> gamma_x_ct(cts_size);
        recv_encrypted_vector(this->context, io, gamma_x_ct);
        auto gamma_x_result = postprocess(gamma_x_ct, data, false);

        saveMatrix("/home/qipang/mnt/d1/linear/EzPC/SCI/build/bin/txt/layernorm_result.txt", gamma_x_result.data(), data.image_size, data.filter_h);
        
        // HACK: verify
        // for (int i = 0; i < 1; i++) {
        //     for (int j = 0; j < data.filter_h; j++) {
        //         cout << gamma_x_result[i * data.filter_h + j] << " ";
        //     }
        //     cout << endl;
        // }

    } else {
        // Server
        #ifdef HE_TIMING
        auto t1_total = high_resolution_clock::now();
        #endif 

        // vector<uint64_t> vec(common_dim);
        // for (int i = 0; i < common_dim; i++) {
        //     vec[i] = B[i][0];
        // }
        vector<vector<uint64_t>> server_sharing(data.image_size, vector<uint64_t>(data.filter_h, 0ULL));

        int cts_size = data.image_size * data.filter_h / data.slot_count;

        auto io_start = io->counter;
        vector<Ciphertext> x1_ct(cts_size);
        recv_encrypted_vector(this->context, io, x1_ct);
        // vector<Ciphertext> x1_ct = {cts.begin(), cts.begin() + cts_size};
        // vector<Ciphertext> var1_ct = {cts.begin() + cts_size, cts.end()};
        vector<Plaintext> gamma_pt = pack_gamma(Gamma, data);
        vector<Plaintext> gamma_x2_pt = gamma_x2_plain_server(X2, Gamma, server_sharing, data);
        vector<Ciphertext> gamma_x_ct = gamma_x_server(x1_ct, gamma_pt, gamma_x2_pt, data);
        send_encrypted_vector(io, gamma_x_ct);
    }
}