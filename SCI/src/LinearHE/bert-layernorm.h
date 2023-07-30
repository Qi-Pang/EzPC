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
};

class LayerNormField {
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

  LayerNormField(int party, NetIO *io);

  ~LayerNormField();

  void configure();

  void print_noise_budget_vec(vector<Ciphertext> v);

  void print_ct(Ciphertext &ct, int len);
  void print_pt(Plaintext &pt, int len);
  void saveMatrix(const std::string& filename, uint64_t* matrix, size_t rows, size_t cols);

  vector<Plaintext> pack_gamma(vector<uint64_t> &gamma, const FCMetadata &data);
  vector<Plaintext> pack_x_plain(vector<vector<uint64_t>> &x, const FCMetadata &data);
  vector<Ciphertext> pack_x_cipher(vector<vector<uint64_t>> &x, const FCMetadata &data);
  vector<Plaintext> gamma_x2_plain_server(vector<vector<uint64_t>> &x2_pt, vector<uint64_t> &gamma, vector<vector<uint64_t>> &sharing_r, const FCMetadata &data);
  vector<Ciphertext> gamma_x_server(vector<Ciphertext> &x1_ct, vector<Plaintext> &enc_gamma, vector<Plaintext> &gamma_x2_pt, const FCMetadata &data);
  vector<vector<uint64_t>> y_var_plain(vector<vector<uint64_t>> &y1, vector<uint64_t> &var1, const FCMetadata &data);
  vector<Plaintext> pack_var_plain(vector<uint64_t> &var, const FCMetadata &data);
  vector<Ciphertext> pack_var_cipher(vector<uint64_t> &var, const FCMetadata &data);
  vector<Ciphertext> y1_var2_server(vector<Ciphertext> &y1_ct, vector<Plaintext> &var2_pt, const FCMetadata &data);
  vector<Ciphertext> var1_y2_server(vector<Ciphertext> &var1_ct, vector<Plaintext> &y2_pt, const FCMetadata &data);
  vector<vector<uint64_t>> x_square_plain(vector<vector<uint64_t>> &x, const FCMetadata &data);
  vector<Ciphertext> x1_x2_server(vector<Ciphertext> &x1, vector<Plaintext> &x2, const FCMetadata &data);
  vector<uint64_t> postprocess(vector<Ciphertext> &cts, const FCMetadata &data, const bool &col_packing = true);

  void layernorm_he(int32_t input_dim, int32_t common_dim,
                            int32_t output_dim,
                            vector<vector<uint64_t>> &X1, 
                            vector<vector<uint64_t>> &X2, 
                            vector<uint64_t> &Gamma, 
                            vector<uint64_t> &Var1, 
                            vector<uint64_t> &Var2);

};
#endif
