/*
Original Author: ryanleh
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

Modified by Deevashwer Rathee
*/

#include "LinearHE/bert-matmul.h"

using namespace std;
using namespace seal;
using namespace sci;

inline std::string uint64_to_hex_string(std::uint64_t value)
{
    return seal::util::uint_to_hex_string(&value, std::size_t(1));
}

void test_fresh_noise(Encryptor &encryptor, Decryptor &decryptor, BatchEncoder &batch_encoder){
    uint64_t x = 6;
    Plaintext x_plain(uint64_to_hex_string(x));

    Ciphertext x_encrypted;
    encryptor.encrypt(x_plain, x_encrypted);
    cout << " ###    + noise budget in freshly encrypted x: " << decryptor.invariant_noise_budget(x_encrypted) << " bits"
         << endl;

    std::vector<uint64_t> pod_matrix(8192, 0ULL);
    Plaintext tmp;
    Ciphertext ciphertext;
    for(int i = 0; i < 8192; i++){
        pod_matrix[i] = i + 2;
    }
    cout << "pod_matrix length: " << pod_matrix.size() << " " << pod_matrix[0] << " " << pod_matrix[1] << endl;
    batch_encoder.encode(pod_matrix, tmp);
    encryptor.encrypt(tmp, ciphertext);
    cout << " ###    + noise budget in freshly encrypted vec: " << decryptor.invariant_noise_budget(ciphertext) << " bits" << endl;

}

vector<Ciphertext> bert_preprocess_vec(const vector<uint64_t> input, const FCMetadata &data,
                          Encryptor &encryptor, Decryptor &decryptor, BatchEncoder &batch_encoder) {
  // Column-wise packing
  vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
//   uint64_t size_pow2 = next_pow2(data.image_size);
//   for (int col = 0; col < data.image_size; col++) {
//     for (int idx = 0; idx < data.pack_num; idx++) {
//       pod_matrix[col + size_pow2 * idx] = input[col];
//     }
//   }
  cout << "Vector Dimension: " << data.image_size << " Vector Packnum: " << data.pack_num << endl;

  vector<Ciphertext> cts;
  for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
  {
    // pod_matrix = vector<uint64_t>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
    for(int i = 0; i < 8192; i++){
        pod_matrix[i] = i;
    }
    cout << "pod_matrix length: " << pod_matrix.size() << " " << pod_matrix[0] << " " << pod_matrix[1] << endl;
    Ciphertext ciphertext;
    Plaintext tmp;
    batch_encoder.encode(pod_matrix, tmp);
    encryptor.encrypt(tmp, ciphertext);
    if(i == 0){
        cout << " ###    + noise budget in freshly encrypted vec: " << decryptor.invariant_noise_budget(ciphertext) << " bits"
         << endl;
    }
    cts.push_back(ciphertext);
  }
  
  return cts;
}

vector<vector<Plaintext>> bert_preprocess_matrix(const uint64_t *const *matrix,
                                    const FCMetadata &data,
                                    BatchEncoder &batch_encoder) {
  // Pack the filter in alternating order of needed ciphertexts. This way we
  // rotate the input once per ciphertext

    vector<vector<Plaintext>> weightMatrix;
    vector<int64_t> temp2;
    int num_diag = 32;
    int num_matrix_per_diag = data.filter_h / (data.slot_count / data.image_size); // should be 12
    for (int l = 0; l < num_diag; l++)
    {//iterate over all diagonals
        vector<Plaintext> temp_matrix_diag(num_matrix_per_diag);
        int matrix_diag_index = 0;
        for (int i = 0; i < data.filter_h / num_diag; i++)
        {//iterate over subblocks (32x32) of rows
            for (int j = 0; j < num_diag; j++) 
            {//iterate over columns
                for (int k = 0; k < 128; k++)
                {
                    temp2.push_back(matrix[i * num_diag + j][(j + l) % num_diag]);
                }
                if (temp2.size() % (data.slot_count / 2) == 0)
                {
                    std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * num_diag, temp2.begin() + temp2.size() - l * 128, temp2.end());
                    // temp1.push_back(cryptoContext->MakePackedPlaintext(temp2));
                    if (temp2.size() == data.slot_count)
                    {
                        batch_encoder.encode(temp2, temp_matrix_diag[matrix_diag_index]);
                        matrix_diag_index++;
                        temp2.clear();
                    }
                }
            }
        }
        weightMatrix.push_back(temp_matrix_diag);
    }

    for (int l = 0; l < num_diag; l++)
    {//iterate over all diagonals
        vector<Plaintext> temp_matrix_diag(num_matrix_per_diag);
        int matrix_diag_index = 0;
        for (int i = 0; i < data.filter_h / num_diag; i++)
        {//iterate over subblocks (32x32) of rows
            for (int j = 0; j < num_diag; j++) 
            {//iterate over columns
                for (int k = 0; k < 128; k++)
                {
                    temp2.push_back(matrix[i * num_diag + j][(j + l) % num_diag + num_diag]);
                }
                if (temp2.size() % (data.slot_count / 2) == 0)
                {
                    std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * num_diag, temp2.begin() + temp2.size() - l * 128, temp2.end());
                    // temp1.push_back(cryptoContext->MakePackedPlaintext(temp2));
                    if (temp2.size() == data.slot_count)
                    {
                        std::rotate(temp2.begin(), temp2.begin() + temp2.size()/2, temp2.end());
                        batch_encoder.encode(temp2, temp_matrix_diag[matrix_diag_index]);
                        matrix_diag_index++;
                        temp2.clear();
                    }
                }
            }
        }
        weightMatrix.push_back(temp_matrix_diag);
    }

    return weightMatrix;


//   vector<vector<uint64_t>> mat_pack(data.inp_ct,
//                                     vector<uint64_t>(data.slot_count, 0ULL));
//   for (int row = 0; row < data.filter_h; row++) {
//     int ct_idx = row / data.inp_ct;
//     for (int col = 0; col < data.filter_w; col++) {
//       mat_pack[row % data.inp_ct][col + next_pow2(data.filter_w) * ct_idx] =
//           matrix[row][col];
//     }
//   }

//   // Take the packed ciphertexts above and repack them in a diagonal ordering.
//   int mod_mask = (data.inp_ct - 1);
//   int wrap_thresh = min(data.slot_count >> 1, next_pow2(data.filter_w));
//   int wrap_mask = wrap_thresh - 1;
//   vector<vector<uint64_t>> mat_diag(data.inp_ct,
//                                     vector<uint64_t>(data.slot_count, 0ULL));
//   for (int ct = 0; ct < data.inp_ct; ct++) {
//     for (int col = 0; col < data.slot_count; col++) {
//       int ct_diag_l = (col - ct) & wrap_mask & mod_mask;
//       int ct_diag_h = (col ^ ct) & (data.slot_count / 2) & mod_mask;
//       int ct_diag = (ct_diag_h + ct_diag_l);

//       int col_diag_l = (col - ct_diag_l) & wrap_mask;
//       int col_diag_h = wrap_thresh * (col / wrap_thresh) ^ ct_diag_h;
//       int col_diag = col_diag_h + col_diag_l;

//       mat_diag[ct_diag][col_diag] = mat_pack[ct][col];
//     }
//   }

//   vector<Plaintext> enc_mat(data.inp_ct);
//   for (int ct = 0; ct < data.inp_ct; ct++) {
//     batch_encoder.encode(mat_diag[ct], enc_mat[ct]);
//   }
//   return enc_mat;
}

/* Generates a masking vector of random noise that will be applied to parts of
 * the ciphertext that contain leakage */
Ciphertext bertfc_preprocess_noise(const uint64_t *secret_share,
                               const FCMetadata &data, Encryptor &encryptor,
                               BatchEncoder &batch_encoder) {
  // Sample randomness into vector
  vector<uint64_t> noise(data.slot_count, 0ULL);
//   PRG128 prg;
//   prg.random_mod_p<uint64_t>(noise.data(), data.slot_count, prime_mod);

  // Puncture the vector with secret shares where an actual fc result value
  // lives
//   for (int row = 0; row < data.filter_h; row++) {
//     int curr_set = row / data.inp_ct;
//     noise[(row % data.inp_ct) + next_pow2(data.image_size) * curr_set] =
//         secret_share[row];
//   }
  for (int i = 0; i < data.slot_count; i++)
    noise[i] = secret_share[i];

  Ciphertext enc_noise;
  Plaintext tmp;
  batch_encoder.encode(noise, tmp);
  encryptor.encrypt(tmp, enc_noise);

  return enc_noise;
}

Ciphertext bertfc_online(vector<Ciphertext> &cts, vector<vector<Plaintext>> &enc_mat,
                     const FCMetadata &data, Evaluator &evaluator,
                     GaloisKeys &gal_keys, Ciphertext &zero,
                     Ciphertext &enc_noise) {

    Ciphertext result = zero;
    //prepare rotated intermediate representation
    vector<vector<Ciphertext>> rotatedIR;
    vector<vector<Ciphertext>> rotatedIR_col;
    int num_diag = 32;
    cout << "[Server] Online Start" << endl;
    int rotation_num_count = 0;
    for (int i = 0; i < cts.size(); i++)
    {   
        vector<Ciphertext> tmp;
        vector<Ciphertext> tmp_col;
        tmp.push_back(cts[i]);
        Ciphertext ct_rotate_col;
        evaluator.rotate_columns(cts[i], gal_keys, ct_rotate_col);
        tmp_col.push_back(ct_rotate_col);
        Ciphertext temp_rot;
        Ciphertext temp_rot_col;
        for (int j = 1; j < num_diag; j++)
        {
            evaluator.rotate_rows(cts[i], (num_diag - j) * 128, gal_keys, temp_rot);
            evaluator.rotate_rows(ct_rotate_col, (num_diag - j) * 128, gal_keys, temp_rot_col);
            rotation_num_count += 1;
            tmp.push_back(temp_rot);
            tmp_col.push_back(temp_rot_col);
        }
        rotatedIR.push_back(tmp);
        rotatedIR_col.push_back(tmp_col);
        tmp.clear();
        tmp_col.clear();
        cout << "[Server] Rotation Num: " << rotation_num_count << endl;
    }
    cout << "[Server] Online Start - rotation done" << endl;
    //compute matrix multiplication
    for (int j = 0; j < enc_mat.size() / 2; j++) {//iterate over all diagonals
        for (int i = 0; i < cts.size(); i++)
        {//iterate over all ciphertexts
            //multiply each diagonal with resp. ciphertext
            Ciphertext temp = rotatedIR[i][j];
            evaluator.multiply_plain_inplace(temp, enc_mat[j][i]);
            // Ciphertext temp = cryptoContext->EvalMult(rotatedIR[i][j],  model_weights[j][i]);
            if (i == 0 && j==0) result = temp;
            else evaluator.add_inplace(result, temp);
        }
    }
    cout << "[Server] Online Start - comp-1 done" << endl;
    for (int j = enc_mat.size() / 2; j < enc_mat.size(); j++) {//iterate over all diagonals
        for (int i = 0; i < cts.size(); i++)
        {//iterate over all ciphertexts
            //multiply each diagonal with resp. ciphertext
            Ciphertext temp = rotatedIR_col[i][j - enc_mat.size()/2];
            evaluator.multiply_plain_inplace(temp, enc_mat[j][i]);
            // Ciphertext temp = cryptoContext->EvalMult(rotatedIR[i][j],  model_weights[j][i]);
            evaluator.add_inplace(result, temp);
        }
    }

    cout << "[Server] Online Done" << endl;

    evaluator.mod_switch_to_next_inplace(result);
    evaluator.mod_switch_to_next_inplace(enc_noise);
    evaluator.add_inplace(result, enc_noise);

    return result;
}

uint64_t *bertfc_postprocess(Ciphertext &ct, const FCMetadata &data,
                         BatchEncoder &batch_encoder, Decryptor &decryptor) {
  vector<uint64_t> plain(data.slot_count, 0ULL);
  Plaintext tmp;
  decryptor.decrypt(ct, tmp);
  batch_encoder.decode(tmp, plain);

  uint64_t *result = new uint64_t[data.image_size*data.filter_w];
  for (int row = 0; row < data.image_size*data.filter_w; row++) {
    result[row] = plain[row];
  }
  return result;
}

BERTFCField::BERTFCField(int party, NetIO *io) {
  this->party = party;
  this->io = io;
  this->slot_count = POLY_MOD_DEGREE;
  generate_new_keys(party, io, slot_count, context, encryptor, decryptor,
                    evaluator, encoder, gal_keys, relin_keys, zero);
}

BERTFCField::~BERTFCField() {
  free_keys(party, encryptor, decryptor, evaluator, encoder, gal_keys, zero);
}

void BERTFCField::configure() {
  data.slot_count = slot_count;
  // Only works with a ciphertext that fits in a single ciphertext
  assert(data.slot_count >= data.image_size);

  data.filter_size = data.filter_h * data.filter_w;
  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / next_pow2(data.filter_w);
  // How many total ciphertexts we'll need
  data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
}

vector<uint64_t> BERTFCField::ideal_functionality(uint64_t *vec,
                                              uint64_t **matrix) {
  vector<uint64_t> result(data.filter_h, 0ULL);
  for (int row = 0; row < data.filter_h; row++) {
    for (int idx = 0; idx < data.filter_w; idx++) {
      uint64_t partial = vec[idx] * matrix[row][idx];
      result[row] = result[row] + partial;
    }
  }
  return result;
}

void BERTFCField::matrix_multiplication(int32_t input_dim, int32_t common_dim,
                                    int32_t output_dim,
                                    vector<vector<uint64_t>> &A,
                                    vector<vector<uint64_t>> &B,
                                    vector<vector<uint64_t>> &C,
                                    bool verify_output, bool verbose) {
  data.filter_h = common_dim;
  data.filter_w = output_dim;
  data.image_size = input_dim;
  this->slot_count =
      min(max(8192, 2 * next_pow2(common_dim)), SEAL_POLY_MOD_DEGREE_MAX);
  configure();

  seal::SEALContext *context_;
  Encryptor *encryptor_;
  Decryptor *decryptor_;
  Evaluator *evaluator_;
  BatchEncoder *encoder_;
  GaloisKeys gal_keys_;
  RelinKeys relin_keys_;
  Ciphertext *zero_;
  if (slot_count > POLY_MOD_DEGREE) {
    // generate_new_keys(party, io, slot_count, context_, encryptor_, decryptor_,
    //                   evaluator_, encoder_, gal_keys_, zero_, relin_keys_);
  } else {
    context_ = this->context;
    // encryptor_ = this->encryptor;
    // decryptor_ = this->decryptor;
    // evaluator_ = this->evaluator;
    // encoder_ = this->encoder;
    // gal_keys_ = this->gal_keys;
    encoder_ = new BatchEncoder(*context_);
    evaluator_ = new Evaluator(*context_);
    KeyGenerator keygen(*context_);
    SecretKey sec_key = keygen.secret_key();
    PublicKey pub_key;
    keygen.create_public_key(pub_key);
    GaloisKeys gal_keys_;
    keygen.create_galois_keys(gal_keys_);
    RelinKeys relin_keys_;
    keygen.create_relin_keys(relin_keys_);

    encryptor_ = new Encryptor(*context_, pub_key);
    decryptor_ = new Decryptor(*context_, sec_key);
    test_fresh_noise(*encryptor_, *decryptor_, *encoder_);
    zero_ = this->zero;
  }

//   return;

  cout << "Key generated" << endl;

  if (party == BOB) {  // Client
    vector<uint64_t> vec(common_dim * input_dim);
    for (int j = 0; j < common_dim; j++)
        for (int i = 0; i < input_dim; i++)
            vec[j*input_dim + i] = A[i][j];

    if (verbose)
      cout << "[Client] Vector Generated" << endl;

    auto cts = bert_preprocess_vec(vec, data, *encryptor_, *decryptor_, *encoder_);

    // FIXME: Debug here

    vector<uint64_t> plain(data.slot_count, 0ULL);
    Plaintext tmp;
    decryptor_->decrypt(cts[0], tmp);
    encoder_->decode(tmp, plain);

    // KeyGenerator keygen(context_);
    // RelinKeys relin_keys;
    // relin_keys = keygen.relin_keys();

    if (verbose)
    {
      cout << "[Client] Debug " << endl;
      for (int i = 0; i < 5; i++)
      {
        cout << "[Client] Debug Noise budget: " << decryptor_->invariant_noise_budget(cts[0]) << " Size: " << cts[0].size() << endl;
        evaluator_->multiply_plain_inplace(cts[0], tmp);
        // evaluator_->multiply_inplace(cts[0], cts[0]);
        // evaluator_->square_inplace(cts[0]);
        evaluator_->relinearize_inplace(cts[0], relin_keys_);
        // evaluator_->mod_switch_to_next_inplace(cts[0]);
      }
      cout << "[Client] Debug Noise budget: " << decryptor_->invariant_noise_budget(cts[0]) << " Size: " << cts[0].size() << endl;

      decryptor_->decrypt(cts[0], tmp);
      encoder_->decode(tmp, plain);
      cout << "[Client] Debug decrypt " << plain[0] << endl;
      // for (uint64_t i : plain)
      //   cout << i << " " << endl;
    }

    print_parameters(*context_);

    // send_ciphertext(io, cts);
    auto io_start = io->counter;
    send_encrypted_vector(io, cts);
    // cout << "size of cts (Bytes): " << sizeof(Ciphertext) << " " << sizeof(Ciphertext) * cts.size() << endl;
    if (verbose)
      cout << "[Client] Vector processed and sent" << endl;

    Ciphertext enc_result;
    recv_ciphertext(context_, io, enc_result);
    cout << "[Client] size of cts (Bytes): " << io->counter - io_start << endl;
    auto HE_result = bertfc_postprocess(enc_result, data, *encoder_, *decryptor_);
    if (verbose)
      cout << "[Client] Result received and decrypted" << endl;

    // for (int i = 0; i < num_rows; i++) {
    //   C[i][0] = HE_result[i];
    // }
    // if (verify_output)
    //   verify(&vec, nullptr, C);

    delete[] HE_result;
  } else // party == ALICE // Server
  {
    vector<uint64_t> vec(common_dim);
    for (int i = 0; i < common_dim; i++) {
      vec[i] = B[i][0];
    }
    if (verbose)
      cout << "[Server] Vector Generated" << endl;
    vector<uint64_t *> matrix_mod_p(common_dim);
    vector<uint64_t *> matrix(common_dim);
    for (int i = 0; i < common_dim; i++) {
      matrix_mod_p[i] = new uint64_t[output_dim];
      matrix[i] = new uint64_t[output_dim];
      for (int j = 0; j < output_dim; j++) {
        matrix_mod_p[i][j] = neg_mod((int64_t)B[i][j], (int64_t)prime_mod);
        int64_t val = (int64_t)B[i][j];
        if (val > int64_t(prime_mod/2)) {
          val = val - prime_mod;
        }
        matrix[i][j] = val;
      }
    }
    if (verbose)
      cout << "[Server] Matrix generated" << endl;

    PRG128 prg;
    uint64_t *secret_share = new uint64_t[input_dim*output_dim];
    prg.random_mod_p<uint64_t>(secret_share, input_dim*output_dim, prime_mod);

    Ciphertext enc_noise =
        bertfc_preprocess_noise(secret_share, data, *encryptor_, *encoder_);
    auto encoded_mat = bert_preprocess_matrix(matrix_mod_p.data(), data, *encoder_);
    if (verbose)
      cout << "[Server] Matrix and noise processed" << endl;

    auto io_start = io->counter;
    vector<Ciphertext> cts(12);

    // recv_ciphertext(io, cts);
    recv_encrypted_vector(context_, io, cts);
    if (verbose)
      cout << "[Server] cts received" << endl;

// #ifdef HE_DEBUG
//     PRINT_NOISE_BUDGET(decryptor_, ct, "before FC Online");
// #endif

    auto HE_result = bertfc_online(cts, encoded_mat, data, *evaluator_, gal_keys_,
                               *zero_, enc_noise);

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result, "after FC Online");
#endif

    parms_id_type parms_id = HE_result.parms_id();
    shared_ptr<const SEALContext::ContextData> context_data =
        context_->get_context_data(parms_id);
    flood_ciphertext(HE_result, context_data, SMUDGING_BITLEN);

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result, "after noise flooding");
#endif

    evaluator_->mod_switch_to_next_inplace(HE_result);

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result, "after mod-switch");
#endif

    send_ciphertext(io, HE_result);
    cout << "[Server] size of result (Bytes): " << io->counter - io_start << endl;
    // auto he_result_parms = context_->get_context_data(HE_result.parms_id());
    // cout << "[Server] size of result (Bytes): " << he_result_parms->parms().coeff_modulus().size() << " " << he_result_parms->parms().poly_modulus_degree() << " " << HE_result.save_size(compr_mode_type::none) << endl;
    if (verbose)
      cout << "[Server] Result computed and sent" << endl;

    // auto result = ideal_functionality(vec.data(), matrix.data());

    // for (int i = 0; i < num_rows; i++) {
    //   C[i][0] = neg_mod((int64_t)result[i] - (int64_t)secret_share[i],
    //                     (int64_t)prime_mod);
    // }
    // if (verify_output)
    //   verify(&vec, &matrix, C);

    for (int i = 0; i < common_dim; i++) {
      delete[] matrix_mod_p[i];
      delete[] matrix[i];
    }
    delete[] secret_share;
  }
  // if (slot_count > POLY_MOD_DEGREE) {
  //   free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_,
  //             zero_);
  // }
}

void BERTFCField::verify(vector<uint64_t> *vec, vector<uint64_t *> *matrix,
                     vector<vector<uint64_t>> &C) {
  if (party == BOB) {
    io->send_data(vec->data(), data.filter_w * sizeof(uint64_t));
    io->flush();
    for (int i = 0; i < data.filter_h; i++) {
      io->send_data(C[i].data(), sizeof(uint64_t));
    }
  } else // party == ALICE
  {
    vector<uint64_t> vec_0(data.filter_w);
    io->recv_data(vec_0.data(), data.filter_w * sizeof(uint64_t));
    for (int i = 0; i < data.filter_w; i++) {
      vec_0[i] = (vec_0[i] + (*vec)[i]) % prime_mod;
    }
    auto result = ideal_functionality(vec_0.data(), matrix->data());

    vector<vector<uint64_t>> C_0(data.filter_h);
    for (int i = 0; i < data.filter_h; i++) {
      C_0[i].resize(1);
      io->recv_data(C_0[i].data(), sizeof(uint64_t));
      C_0[i][0] = (C_0[i][0] + C[i][0]) % prime_mod;
    }
    bool pass = true;
    for (int i = 0; i < data.filter_h; i++) {
      if (neg_mod(result[i], (int64_t)prime_mod) != (int64_t)C_0[i][0]) {
        pass = false;
      }
    }
    if (pass)
      cout << GREEN << "[Server] Successful Operation" << RESET << endl;
    else {
      cout << RED << "[Server] Failed Operation" << RESET << endl;
      cout << RED << "WARNING: The implementation assumes that the computation"
           << endl;
      cout << "performed locally by the server (on the model and its input "
              "share)"
           << endl;
      cout << "fits in a 64-bit integer. The failed operation could be a result"
           << endl;
      cout << "of overflowing the bound." << RESET << endl;
    }
  }
}

void print_parameters(const seal::SEALContext &context)
{
    auto &context_data = *context.key_context_data();
    /*
    Which scheme are we using?
    */
    std::string scheme_name = "BFV";
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;
    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;
    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    std::cout << "\\" << std::endl;
}
