#include "LinearHE/bert-layernorm.h"
#include <omp.h>

using namespace std;
using namespace sci;
using namespace seal;

#define HE_TIMING
// #define HE_DEBUG

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
                temp_pack[j + k * data.image_size] = gamma[i * k];
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
            temp_pack[ind] = x[row][col];
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
        cout << "debug " << i << endl;
        vector<uint64_t> temp_pack(data.slot_count, 0ULL);
        for (int ind = 0; ind < data.slot_count; ind++) {
            int row = ind % data.image_size;
            int col = ind / data.image_size + i * data.slot_count / data.image_size;
        }
        Plaintext pt1;
        Ciphertext ct1;
        encoder->encode(temp_pack, pt1);
        cout << "debug encode done " << endl;

        encryptor->encrypt(pt1, ct1);
        result[i] = ct1;
    }
    return result;
}

vector<Plaintext> LayerNormField::gamma_x2_plain_server(vector<vector<uint64_t>> &x2_pt, vector<uint64_t> &gamma, vector<vector<uint64_t>> &sharing_r, const FCMetadata &data) {
    for (int i = 0; i < data.image_size; i++) {
        for (int j = 0; j < data.filter_h; j++) {
            x2_pt[i][j] = x2_pt[i][j] * gamma[j] - sharing_r[i][j];
        }
    }
    return pack_x_plain(x2_pt, data);
}

vector<Ciphertext> LayerNormField::gamma_x_server(vector<Ciphertext> &x1_ct, vector<Plaintext> &enc_gamma, vector<Plaintext> &gamma_x2_pt, const FCMetadata &data) {
    int cts_size = x1_ct.size();
    for (int i = 0; i < cts_size; i++) {
        evaluator->multiply_plain_inplace(x1_ct[i], enc_gamma[i]);
        evaluator->add_plain_inplace(x1_ct[i], gamma_x2_pt[i]);
    }
    return x1_ct;
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
            temp_pack[ind] = var[col];
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

    cout << "debug config done " << endl;

    if (party == BOB) {
        // Client

        vector<Ciphertext> x1_ct = pack_x_cipher(X1, data);
        vector<Ciphertext> var1_ct = pack_var_cipher(Var1, data);
        x1_ct.insert(x1_ct.end(), var1_ct.begin(), var1_ct.end());
        cout << "debug client packing done " << endl;
        send_encrypted_vector(io, x1_ct);

    } else {
        // Server
        #ifdef HE_TIMING
        auto t1_total = high_resolution_clock::now();
        #endif 

        // vector<uint64_t> vec(common_dim);
        // for (int i = 0; i < common_dim; i++) {
        //     vec[i] = B[i][0];
        // }

        int cts_size = data.image_size * data.filter_h / data.slot_count;

        auto io_start = io->counter;
        vector<Ciphertext> cts(cts_size * 2);
        recv_encrypted_vector(this->context, io, cts);
        vector<Ciphertext> x1_ct = {cts.begin(), cts.begin() + cts_size};
        vector<Ciphertext> var1_ct = {cts.begin() + cts_size, cts.end()};
    }
}