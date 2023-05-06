/*
Original Author: ryanleh, Deevashwer Rathee
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

Modified by Qi Pang
*/

#include "LinearHE/bert-matmul-cipher-efficient.h"
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/bfvrns/bfvrns-ser.h"
// #include "scheme/bgvrns/bgvrns-ser.h"
#include <omp.h>

using namespace std;
using namespace sci;
using namespace lbcrypto;

// Generate the masks for 1-step rotation
vector<vector<lbcrypto::Plaintext>> generate_rotation_masks(const FCMetadata &data, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext)
{
    vector<vector<lbcrypto::Plaintext>> result;
    for (int i = 0; i < 128; i++)
    {
        vector<lbcrypto::Plaintext> temp_result(2);
        vector<int64_t> mask1(data.slot_count, 0LL);
        vector<int64_t> mask2(data.slot_count, 0LL);
        for (int j = 0; j < 128 - i; j++)
            for (int k = 0; k < 32; k++)
            {
                mask1[j + 128 * k] = 1;
                mask1[j + 128 * k + data.slot_count / 2] = 1;
            }

        for (int j = 128 - i; j < 128; j++)
            for (int k = 0; k < 32; k++)
            {
                mask2[j + 128 * k] = 1;
                mask2[j + 128 * k + data.slot_count / 2] = 1;
            }
        temp_result[0] = cryptoContext->MakePackedPlaintext(mask1);
        temp_result[1] = cryptoContext->MakePackedPlaintext(mask2);
        result.push_back(temp_result);
    }
    return result;
}

// Generate cipher_masks: 1111100000..., 0000011111..., ...
vector<lbcrypto::Plaintext> generate_cipher_masks(const FCMetadata &data, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext)
{
    vector<lbcrypto::Plaintext> result;
    for (int i = 0; i < 32; i++)
    {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128; k++)
            mask1[i * 128 + k] = 1;
        result.push_back(cryptoContext->MakePackedPlaintext(mask1));
    }

    for (int i = 0; i < 32; i++)
    {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128; k++)
            mask1[i * 128 + k + data.slot_count / 2] = 1;
        result.push_back(cryptoContext->MakePackedPlaintext(mask1));
    }

    for (int i = 0; i < 32; i++)
    {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128; k++)
            mask1[i * 128 + k] = 1;
        result.push_back(cryptoContext->MakePackedPlaintext(mask1));
    }

    for (int i = 0; i < 32; i++)
    {
        vector<int64_t> mask1(data.slot_count, 0LL);
        for (int k = 0; k < 128; k++)
            mask1[i * 128 + k + data.slot_count / 2] = 1;
        result.push_back(cryptoContext->MakePackedPlaintext(mask1));
    }
    return result;
}

// Generate packing_masks: 1111100000, 0000011111
vector<lbcrypto::Plaintext> generate_packing_masks(const FCMetadata &data, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext)
{
    vector<lbcrypto::Plaintext> result;
    vector<int64_t> mask1(data.slot_count, 0LL);
    vector<int64_t> mask2(data.slot_count, 0LL);
    for (int i = 0; i < data.slot_count / 2; i++)
    {
        mask1[i] = 1;
        mask2[i + data.slot_count / 2] = 1;
        
    }
    result.push_back(cryptoContext->MakePackedPlaintext(mask1));
    result.push_back(cryptoContext->MakePackedPlaintext(mask2));

    return result;
}

// 1-step rotations contain 2 sub-rotations
lbcrypto::Ciphertext<DCRTPoly> hoisted_rotation_by_one(const FCMetadata &data, lbcrypto::Ciphertext<DCRTPoly> ct, int k, vector<vector<lbcrypto::Plaintext>> rotation_masks, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext, std::shared_ptr<vector<lbcrypto::DCRTPoly>> &rotation_precompute)
{
    int m = -(32 - k);
    lbcrypto::Ciphertext<DCRTPoly> ct1 = cryptoContext->EvalFastRotation(ct, k, 2 * data.slot_count, rotation_precompute);
    lbcrypto::Ciphertext<DCRTPoly> ct2 = cryptoContext->EvalFastRotation(ct, m, 2 * data.slot_count, rotation_precompute);
    return cryptoContext->EvalAdd(cryptoContext->EvalMult(ct1, rotation_masks[k][0]), cryptoContext->EvalMult(ct2, rotation_masks[k][1]));
}

lbcrypto::Ciphertext<DCRTPoly> rotation_by_one(const FCMetadata &data, lbcrypto::Ciphertext<DCRTPoly> ct, int k, vector<vector<lbcrypto::Plaintext>> rotation_masks, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext)
{
    int m = -(32 - k);
    lbcrypto::Ciphertext<DCRTPoly> ct1 = cryptoContext->EvalRotate(ct, k);
    lbcrypto::Ciphertext<DCRTPoly> ct2 = cryptoContext->EvalRotate(ct, m);
    return cryptoContext->EvalAdd(cryptoContext->EvalMult(ct1, rotation_masks[k][0]), cryptoContext->EvalMult(ct2, rotation_masks[k][1]));
}

// column-wise packing
vector<lbcrypto::Ciphertext<DCRTPoly>> bert_efficient_preprocess_vec(vector<uint64_t> &input, const FCMetadata &data, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext, lbcrypto::KeyPair<DCRTPoly> &keyPair) {

    vector<int64_t> pod_matrix(data.slot_count, 0ULL);
    vector<lbcrypto::Ciphertext<DCRTPoly>> cts;
    for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
    {
        pod_matrix = vector<int64_t>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
        lbcrypto::Ciphertext<DCRTPoly> ciphertext;
        lbcrypto::Plaintext tmp;
        tmp = cryptoContext->MakePackedPlaintext(pod_matrix);
        ciphertext = cryptoContext->Encrypt(keyPair.publicKey, tmp);
        cts.push_back(ciphertext);
    }
    return cts;
}

vector<vector<lbcrypto::Plaintext>> bert_efficient_preprocess_matrix(const uint64_t *const *matrix, const FCMetadata &data, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext) {

    vector<vector<Plaintext>> weightMatrix;
    vector<int64_t> temp2;
    int num_diag = 32;
    int num_matrix_per_diag = data.filter_h / (data.slot_count / data.image_size); // should be 12
    for (int l = 0; l < num_diag; l++)
    {//iterate over all diagonals
        vector<lbcrypto::Plaintext> temp_matrix_diag(num_matrix_per_diag);
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
                    std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - l * 128, temp2.end());
                    if (temp2.size() == data.slot_count)
                    {
                        temp_matrix_diag[matrix_diag_index] = cryptoContext->MakePackedPlaintext(temp2);
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
        vector<lbcrypto::Plaintext> temp_matrix_diag(num_matrix_per_diag);
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
                    std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - l * 128, temp2.end());
                    if (temp2.size() == data.slot_count)
                    {
                        temp_matrix_diag[matrix_diag_index] = cryptoContext->MakePackedPlaintext(temp2);
                        matrix_diag_index++;
                        temp2.clear();
                    }
                }
            }
        }
        weightMatrix.push_back(temp_matrix_diag);
    }
    return weightMatrix;
}

/* Generates a masking vector of random noise that will be applied to parts of
 * the ciphertext that contain leakage */
lbcrypto::Ciphertext<DCRTPoly> bert_efficient_preprocess_noise(const uint64_t *secret_share, const FCMetadata &data, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext, lbcrypto::KeyPair<DCRTPoly> &keyPair) {
  // Sample randomness into vector
  vector<int64_t> noise(data.slot_count, 0ULL);
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

  lbcrypto::Plaintext tmp = cryptoContext->MakePackedPlaintext(noise);
  lbcrypto::Ciphertext<DCRTPoly> enc_noise = cryptoContext->Encrypt(keyPair.publicKey, tmp);

  return enc_noise;
}

std::vector<lbcrypto::Ciphertext<DCRTPoly>> bert_efficient_online(vector<lbcrypto::Ciphertext<DCRTPoly>> &cts, vector<vector<lbcrypto::Plaintext>> &enc_mat1, vector<vector<lbcrypto::Plaintext>> &enc_mat2, const FCMetadata &data, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext, lbcrypto::KeyPair<DCRTPoly> &keyPair, vector<vector<lbcrypto::Plaintext>> & rotation_masks) {

    //prepare rotated intermediate representation
    vector<vector<lbcrypto::Ciphertext<DCRTPoly>>> rotatedIR(cts.size());

    int num_diag = 32;
    cout << "[Server] Online Start" << endl;
    auto t1 = high_resolution_clock::now();
    // omp_set_num_threads(12);
    #pragma omp parallel for
    for (int i = 0; i < cts.size(); i++)
    {   
        vector<lbcrypto::Ciphertext<DCRTPoly>> tmp;
        tmp.push_back(cts[i]);
        auto cPrecomp = cryptoContext->EvalFastRotationPrecompute(cts[i]);

        lbcrypto::Ciphertext<DCRTPoly> temp_rot;
        for (int j = 1; j < num_diag; j++)
        {
            temp_rot = cryptoContext->EvalFastRotation(cts[i], (num_diag - j) * data.image_size, 2 * data.slot_count, cPrecomp);
            tmp.push_back(temp_rot);
        }
        rotatedIR[i] = tmp;
        tmp.clear();
    }

    auto t2 = high_resolution_clock::now();
    auto ms_double = (t2 - t1)/1e+9;
    cout << "[Server] Online - rotation done " << ms_double.count() << endl;
    //compute matrix multiplication
    vector<lbcrypto::Ciphertext<DCRTPoly>> temp_result1(enc_mat1.size() * cts.size());
    vector<lbcrypto::Ciphertext<DCRTPoly>> temp_result2(enc_mat2.size() * cts.size());
    t1 = high_resolution_clock::now();
    // for (int j = 0; j < enc_mat.size() / 2; j++) {//iterate over all diagonals
    //     for (int i = 0; i < cts.size(); i++)
    #pragma omp parallel for
    for (int k = 0; k < enc_mat1.size() / 2 * cts.size(); k++)
    {//iterate over all ciphertexts
        int j = k / cts.size();
        int i = k % cts.size();
        temp_result1[k] = cryptoContext->EvalMult(rotatedIR[i][j],  enc_mat1[j][i]); // left half
        temp_result1[k + enc_mat1.size() * cts.size() / 2] = cryptoContext->EvalMult(rotatedIR[i][j],  enc_mat1[j + enc_mat1.size() / 2][i]); // right half

        temp_result2[k] = cryptoContext->EvalMult(rotatedIR[i][j],  enc_mat2[j][i]); // left half
        temp_result2[k + enc_mat2.size() * cts.size() / 2] = cryptoContext->EvalMult(rotatedIR[i][j],  enc_mat2[j + enc_mat2.size() / 2][i]); // right half
    }

    lbcrypto::Ciphertext<DCRTPoly> result_left1 = temp_result1[0];
    lbcrypto::Ciphertext<DCRTPoly> result_left2 = temp_result2[0];

    for (int k = 1; k < temp_result1.size() / 2; k++)
    {
        result_left1 = cryptoContext->EvalAdd(temp_result1[k], result_left1);
        result_left2 = cryptoContext->EvalAdd(temp_result2[k], result_left2);
    }
    result_left1 = cryptoContext->EvalAdd(cryptoContext->EvalRotate(result_left1, data.slot_count/2), result_left1);
    result_left2 = cryptoContext->EvalAdd(cryptoContext->EvalRotate(result_left2, data.slot_count/2), result_left2);

    lbcrypto::Ciphertext<DCRTPoly> result_right1 = temp_result1[temp_result1.size() / 2];
    lbcrypto::Ciphertext<DCRTPoly> result_right2 = temp_result2[temp_result1.size() / 2];
    for (int k = 1; k < temp_result1.size() / 2; k++)
    {
        result_right1 = cryptoContext->EvalAdd(temp_result1[k + temp_result1.size() / 2], result_right1);
        result_right2 = cryptoContext->EvalAdd(temp_result2[k + temp_result2.size() / 2], result_right2);
    }
    result_right1 = cryptoContext->EvalAdd(cryptoContext->EvalRotate(result_right1, data.slot_count/2), result_right1);
    result_right2 = cryptoContext->EvalAdd(cryptoContext->EvalRotate(result_right2, data.slot_count/2), result_right2);

    // result_left1 = cryptoContext->EvalMult(result_left1, packing_masks[0]);
    // result_right1 = cryptoContext->EvalMult(result_right1, packing_masks[1]);
    // result = cryptoContext->EvalAdd(result_left1, result_right1);

    vector<lbcrypto::Ciphertext<DCRTPoly>> result = {result_left1, result_right1, result_left2, result_right2};

    t2 = high_resolution_clock::now();
    ms_double = (t2 - t1)/1e+9;
    cout << "[Server] Online Done " << ms_double.count() << endl;

    return result;
}

// 1. rotate rhs for 128 x 1-step rotations
// 2. mult with lhs (producing 128 cts)
// 3. for each of the 128 cts, rotate for log(32) times, sum together + 1 time batch rotation
// 4. mult masks (1, 0 (x31), 1, 0 (x31), ... ) and sum together (do the first 32 (1st batch) and then the second batch).

vector<lbcrypto::Ciphertext<DCRTPoly>> bert_efficient_cipher(const FCMetadata &data, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext, lbcrypto::KeyPair<DCRTPoly> &keyPair, vector<lbcrypto::Ciphertext<DCRTPoly>> Cipher_plain_result, vector<vector<lbcrypto::Plaintext>>& rotation_masks, vector<lbcrypto::Plaintext>& cipher_masks)
{
    auto HE_result1_left = Cipher_plain_result[0];
    auto HE_result1_right = Cipher_plain_result[1];
    auto HE_result2_left = Cipher_plain_result[2];
    auto HE_result2_right = Cipher_plain_result[3];
    vector<lbcrypto::Ciphertext<DCRTPoly>> rotation_results(data.image_size);
    auto t1 = high_resolution_clock::now();
    vector<lbcrypto::Ciphertext<DCRTPoly>> rotation_results_left(data.image_size);
    vector<lbcrypto::Ciphertext<DCRTPoly>> rotation_results_right(data.image_size);
    auto cPrecomp_left = cryptoContext->EvalFastRotationPrecompute(HE_result2_left);
    auto cPrecomp_right = cryptoContext->EvalFastRotationPrecompute(HE_result2_right);
    // lbcrypto::Ciphertext<DCRTPoly> temp_mult = cryptoContext->EvalFastRotation(HE_result2, 2, 2 * data.slot_count, cPrecomp);
    int num_diag = 32;
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++)
    {
        lbcrypto::Ciphertext<DCRTPoly> temp_mult = hoisted_rotation_by_one(data, HE_result2_left, i, rotation_masks, cryptoContext, cPrecomp_left);
        rotation_results_left[i] = cryptoContext->EvalMult(HE_result1_left, temp_mult);
        temp_mult = hoisted_rotation_by_one(data, HE_result2_right, i, rotation_masks, cryptoContext, cPrecomp_right);
        rotation_results_right[i] = cryptoContext->EvalMult(HE_result1_right, temp_mult);
        rotation_results[i] = cryptoContext->EvalAdd(rotation_results_left[i], rotation_results_right[i]);
    }
    auto t2 = high_resolution_clock::now();
    auto ms_double = (t2 - t1)/1e+9;
    std::cout << "[Server] Cipher-Cipher Rotation 1 " << ms_double.count() << std::endl;

    t1 = high_resolution_clock::now();
    int local_rotation = std::ceil(std::log2(32));
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++)
    {
        // rotation_results_left[i] = cryptoContext->EvalAdd(cryptoContext->EvalRotate(rotation_results_left[i], data.slot_count/2), rotation_results_left[i]);

        // auto cPrecomp = cryptoContext->EvalFastRotationPrecompute(rotation_results[i]);
        // Ciphertext<DCRTPoly> temp_rotation = rotation_results[i];
        for (int k = 0; k < local_rotation; k++)
        {
            Ciphertext<DCRTPoly> temp2 = cryptoContext->EvalRotate(rotation_results[i], (int32_t) pow(2, k) * 128);
            rotation_results[i] = cryptoContext->EvalAdd(temp2, rotation_results[i]);
        }
        // Plaintext plaintextMultResult;
        // cryptoContext->Decrypt(keyPair.secretKey, rotation_results[i], &plaintextMultResult);
        // std::cout << "Dec Result rotation results 0: " << plaintextMultResult << endl;

        rotation_results[i] = cryptoContext->EvalMult(rotation_results[i], cipher_masks[i]);

        // cryptoContext->Decrypt(keyPair.secretKey, rotation_results[i], &plaintextMultResult);
        // std::cout << "Dec Result rotation results 0: " << plaintextMultResult << endl;
    }
    t2 = high_resolution_clock::now();
    ms_double = (t2 - t1)/1e+9;
    std::cout << "[Server] Cipher-Cipher Rotation 2 " << ms_double.count() << std::endl;
    // Packing
    t1 = high_resolution_clock::now();
    vector<lbcrypto::Ciphertext<DCRTPoly>> results(2);
    results[0] = rotation_results[0];
    results[1] = rotation_results[data.slot_count / data.image_size];

    for (int i = 1; i < data.slot_count / data.image_size; i++)
    {
        results[0] = cryptoContext->EvalAdd(results[0], rotation_results[i]);
        results[1] = cryptoContext->EvalAdd(results[1], rotation_results[i + data.slot_count / data.image_size]);
    }

    t2 = high_resolution_clock::now();
    ms_double = (t2 - t1)/1e+9;
    std::cout << "[Server] Cipher-Cipher Packing " << ms_double.count() << std::endl;

    return results;
}

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

uint64_t *bert_efficient_postprocess(vector<lbcrypto::Ciphertext<DCRTPoly>> &cts, const FCMetadata &data, lbcrypto::CryptoContext<DCRTPoly> &cryptoContext, lbcrypto::KeyPair<DCRTPoly> &keyPair) {
  uint64_t *result = new uint64_t[data.image_size*data.image_size];
  for (int i = 0; i < cts.size(); i++)
  {
    vector<int64_t> plain(data.slot_count, 0ULL);
    lbcrypto::Plaintext tmp;
    cryptoContext->Decrypt(keyPair.secretKey, cts[i], &tmp);
    //   decryptor.decrypt(ct, tmp);
    //   batch_encoder.decode(tmp, plain);
    plain = tmp->GetPackedValue();
    #pragma omp parallel for
    for (int row = 0; row < data.slot_count; row++) {
        int j = row / 32; // row num
        int k = row % 32; // col num
        if (row >= data.slot_count / 2)
        {
            j = j - data.slot_count / 2;
            k = k + 32;
        }
        result[i * data.slot_count + k * data.image_size + j] = plain[row];
    }
  }
  return result;
}

BERTEFFICIENTFCFieldOPENFHE::BERTEFFICIENTFCFieldOPENFHE(int party, NetIO *io) {
    this->party = party;
    this->io = io;
    this->slot_count = 8192;

    //   generate_new_keys(party, io, slot_count, context, encryptor, decryptor,
                        // evaluator, encoder, gal_keys, zero);
    // this->parameters.SetPlaintextModulus(4293918721);
    this->parameters.SetPlaintextModulus(268582913);
    // this->parameters.SetPlaintextModulus(65537);
    // this->parameters.SetPlaintextModulus(114689);
    this->parameters.SetMultiplicativeDepth(4);
    this->parameters.SetMaxRelinSkDeg(2);
    this->parameters.SetSecurityLevel(HEStd_128_classic);
    // this->parameters.SetScalingTechnique(FIXEDMANUAL);
    this->parameters.SetKeySwitchTechnique(BV);
    this->parameters.SetMultiplicationTechnique(BEHZ);
    this->parameters.SetFirstModSize(44);
    this->parameters.SetRingDim(8192);
    cout << this->parameters.GetFirstModSize() << endl;
    // this->parameters.SetBatchSize(8192);
    
    /*
    this->cryptoContext = GenCryptoContext(this->parameters);
    this->cryptoContext->Enable(PKE);
    // this->cryptoContext->Enable(KEYSWITCH);
    this->cryptoContext->Enable(LEVELEDSHE);
    this->cryptoContext->Enable(ADVANCEDSHE);
    this->keyPair = this->cryptoContext->KeyGen();
    this->cryptoContext->EvalMultKeyGen(this->keyPair.secretKey);
    std::vector<int> v(256 + 33);
    // std::iota(v.begin(), v.end(), 1);
    // for (size_t i = 0; i < 14; i++)
    //     v[i] = (int32_t) pow(2, i);
    for (size_t i = 0; i < 256; i++)
        v[i] = i - 128;
    for (size_t i = 0; i <= 32; i++)
        v[i + 256] = i * 128;
    this->cryptoContext->EvalRotateKeyGen(this->keyPair.secretKey, v);
    std::for_each(v.begin(), v.end(), [](int& d) { d+=4096;});
    this->cryptoContext->EvalRotateKeyGen(this->keyPair.secretKey, v);
    this->M = 4 * (this->cryptoContext->GetRingDimension());
    cout << "Ring Dimension: " << this->cryptoContext->GetRingDimension();
    cout << "Key generated " << this->keyPair.publicKey << endl;
    if (party == BOB)
    {
        Serial::SerializeToFile("./cryptocontext.txt", this->cryptoContext, SerType::JSON);
        Serial::SerializeToFile("./publickey.txt", this->keyPair.publicKey, SerType::JSON);
        Serial::SerializeToFile("./secretkey.txt", this->keyPair.secretKey, SerType::JSON);
        cout << "[BOB] Key Loaded " << this->keyPair.publicKey << endl;
        lbcrypto::CryptoContext<DCRTPoly> new_cryptoContext;
        lbcrypto::KeyPair<DCRTPoly> new_keyPair;
        Serial::DeserializeFromFile("./cryptocontext.txt", new_cryptoContext, SerType::JSON);
        Serial::DeserializeFromFile("./publickey.txt", new_keyPair.publicKey, SerType::JSON);
        Serial::DeserializeFromFile("./secretkey.txt", new_keyPair.secretKey, SerType::JSON);
        std::ofstream emkeyfile("./evalmultkey.txt", std::ios::out | std::ios::binary);
        this->cryptoContext->SerializeEvalMultKey(emkeyfile, SerType::BINARY);
        emkeyfile.close();

        std::ifstream emkeys("./evalmultkey.txt", std::ios::in | std::ios::binary);
        new_cryptoContext->DeserializeEvalMultKey(emkeys, SerType::BINARY);
        emkeys.close();

        std::ofstream emkeyrotfile("./evalrotkey.txt", std::ios::out | std::ios::binary);
        this->cryptoContext->SerializeEvalAutomorphismKey(emkeyrotfile, SerType::BINARY);
        emkeyrotfile.close();

        std::ifstream emrotkeys("./evalrotkey.txt", std::ios::in | std::ios::binary);
        new_cryptoContext->DeserializeEvalAutomorphismKey(emrotkeys, SerType::BINARY);
        emrotkeys.close();

        cout << party << " Key Loaded " << new_keyPair.publicKey << endl;
        if (*(new_keyPair.publicKey) == *(this->keyPair.publicKey))
          cout << " the same " << endl;
    }
    */
    // /*
    Serial::DeserializeFromFile("./cryptocontext.txt", this->cryptoContext, SerType::JSON);
    Serial::DeserializeFromFile("./publickey.txt", this->keyPair.publicKey, SerType::JSON);
    Serial::DeserializeFromFile("./secretkey.txt", this->keyPair.secretKey, SerType::JSON);
    std::ifstream emrotkeys("./evalrotkey.txt", std::ios::in | std::ios::binary);
    this->cryptoContext->DeserializeEvalAutomorphismKey(emrotkeys, SerType::BINARY);
    emrotkeys.close();
    std::ifstream emkeys("./evalmultkey.txt", std::ios::in | std::ios::binary);
    this->cryptoContext->DeserializeEvalMultKey(emkeys, SerType::BINARY);
    emkeys.close();
    cout << party << " Key Loaded " << this->keyPair.publicKey << endl;
    std::cout << "log2 q = "
              << log2(cryptoContext->GetCryptoParameters()->GetElementParams()->GetModulus().ConvertToDouble())
              << std::endl;
    cout << "Ring Dimension: " << this->cryptoContext->GetRingDimension() << endl;
    // */
}

BERTEFFICIENTFCFieldOPENFHE::~BERTEFFICIENTFCFieldOPENFHE() {
//   free_keys(party, encryptor, decryptor, evaluator, encoder, gal_keys, zero);
}

void BERTEFFICIENTFCFieldOPENFHE::configure() {
  data.slot_count = 8192;
  // Only works with a ciphertext that fits in a single ciphertext
  assert(data.slot_count >= data.image_size);

  data.filter_size = data.filter_h * data.filter_w;
  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / next_pow2(data.filter_w);
  // How many total ciphertexts we'll need
  data.inp_ct = ceil((float)next_pow2(data.filter_h) / data.pack_num);
}

vector<uint64_t> BERTEFFICIENTFCFieldOPENFHE::ideal_functionality(uint64_t *vec,
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

void BERTEFFICIENTFCFieldOPENFHE::matrix_multiplication(int32_t input_dim, int32_t common_dim, int32_t output_dim, vector<vector<uint64_t>> &A, vector<vector<uint64_t>> &B, vector<vector<uint64_t>> &C, bool verify_output, bool verbose) {

    cout << "Calling BERT MATMUL" << endl;
    data.filter_h = common_dim;
    data.filter_w = output_dim;
    data.image_size = input_dim;
    this->slot_count = 8192;
    configure();

    lbcrypto::CCParams<CryptoContextBFVRNS> parameters_ = this->parameters;
    lbcrypto::KeyPair<DCRTPoly> keyPair_ = this->keyPair;
    lbcrypto::CryptoContext<DCRTPoly> cryptoContext_ = this->cryptoContext;
    uint32_t M_;

    if (party == BOB) {  // Client
    
    vector<uint64_t> vec(common_dim * input_dim);
    for (int j = 0; j < common_dim; j++)
        for (int i = 0; i < input_dim; i++)
            vec[j*input_dim + i] = A[i][j];

    if (verbose)
        cout << "[Client] Vector Generated" << endl;
    auto cts = bert_efficient_preprocess_vec(vec, data, cryptoContext_, keyPair_);
    // send_ciphertext(io, cts);
    auto io_start = io->counter;
    send_encrypted_vector_openfhe(io, cts);
    // cout << "size of cts (Bytes): " << sizeof(Ciphertext) << " " << sizeof(Ciphertext) * cts.size() << endl;
    if (verbose)
        cout << "[Client] Vector processed and sent" << endl;

    vector<lbcrypto::Ciphertext<DCRTPoly>> enc_result(2);
    recv_encrypted_vector_openfhe(io, enc_result);
    cout << "[Client] size of cts (Bytes): " << io->counter - io_start << endl;
    auto HE_result = bert_efficient_postprocess(enc_result, data, cryptoContext_, keyPair_);
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
    auto t1 = high_resolution_clock::now();
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
      cout << "[Server] Matrix generated with prime mod: " << prime_mod << endl;

    PRG128 prg;
    uint64_t *secret_share = new uint64_t[input_dim*output_dim];
    prg.random_mod_p<uint64_t>(secret_share, input_dim*output_dim, prime_mod);
    auto encoded_mat = bert_efficient_preprocess_matrix(matrix_mod_p.data(), data, cryptoContext_);

    if (verbose)
      cout << "[Server] Matrix processed" << endl;

    auto rotation_masks = generate_rotation_masks(data, cryptoContext_);
    auto cipher_masks = generate_cipher_masks(data, cryptoContext_);
    // auto packing_masks = generate_packing_masks(data, cryptoContext_);

    if (verbose)
      cout << "[Server] Masks processed" << endl;

    // lbcrypto::Ciphertext<DCRTPoly> enc_noise = bert_efficient_preprocess_noise(secret_share, data, cryptoContext_, keyPair_);
    // if (verbose)
    //   cout << "[Server] Noise processed" << endl;

    auto t2 = high_resolution_clock::now();
    auto ms_double = (t2 - t1)/1e+9;
    if (verbose)
      cout << "[Server] Matrix processed " << ms_double.count() << "sec" << endl;

    auto io_start = io->counter;
    vector<lbcrypto::Ciphertext<DCRTPoly>> cts(12);
    recv_encrypted_vector_openfhe(io, cts);

    t2 = high_resolution_clock::now();
    ms_double = (t2 - t1)/1e+9;
    if (verbose)
      cout << "[Server] cts received " << ms_double.count() << endl;

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, ct, "before FC Online");
#endif

    lbcrypto::Ciphertext<DCRTPoly> HE_result1;
    lbcrypto::Ciphertext<DCRTPoly> HE_result2;
    // omp_set_num_threads(4);
    // #pragma omp parallel private(HE_result1, HE_result2) shared(cts, encoded_mat, data, cryptoContext_, keyPair_) 
    // {
    //     #pragma omp single 
    //     {
    //     #pragma omp task
    //         bertfc_cipher_online(cts, encoded_mat, data, cryptoContext_, keyPair_, HE_result1);
    //     #pragma omp task
    //         bertfc_online_col(cts, encoded_mat_col_pack, data, cryptoContext_, keyPair_, ciphermasks_col[0], ciphermasks_col[1], HE_result2);
    //     }
    // }
    auto Cipher_plain_results = bert_efficient_online(cts, encoded_mat, encoded_mat, data, cryptoContext_, keyPair_, rotation_masks);

    auto t3 = high_resolution_clock::now();
    ms_double = (t3 - t1)/1e+9;
    if (verbose)
      cout << "[Server] Cipher-Plaintext Matmul Done " << ms_double.count() << "sec" << endl;

    // Plaintext plaintextMultResult;
    // cryptoContext->Decrypt(keyPair.secretKey, HE_result1, &plaintextMultResult);
    // auto reencrypt_vector = plaintextMultResult->GetPackedValue();
    // HE_result1 = cryptoContext->Encrypt(keyPair_.publicKey, cryptoContext_->MakePackedPlaintext(reencrypt_vector));

    // cryptoContext->Decrypt(keyPair.secretKey, HE_result2, &plaintextMultResult);
    // reencrypt_vector = plaintextMultResult->GetPackedValue();
    // HE_result2 = cryptoContext->Encrypt(keyPair_.publicKey, cryptoContext_->MakePackedPlaintext(reencrypt_vector));

    // if (verbose)
    //   cout << "[Server] Reencrypted " << endl;

    auto HE_result = bert_efficient_cipher(data, cryptoContext_, keyPair_, Cipher_plain_results, rotation_masks, cipher_masks);

    auto t5 = high_resolution_clock::now();
    ms_double = (t5 - t1)/1e+9;
    if (verbose)
      cout << "[Server] Cipher-Cipher Matmul Done " << ms_double.count() << "sec" << endl;

    Plaintext plaintextMultResult;
    cryptoContext->Decrypt(keyPair.secretKey, HE_result[0], &plaintextMultResult);
    cout << plaintextMultResult << endl;

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result, "after FC Online");
#endif

    // FIXME: 
    // parms_id_type parms_id = HE_result.parms_id();
    // shared_ptr<const SEALContext::ContextData> context_data =
    //     context_->get_context_data(parms_id);
    // flood_ciphertext(HE_result, context_data, SMUDGING_BITLEN);

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result, "after noise flooding");
#endif

    // FIXME:
    // evaluator_->mod_switch_to_next_inplace(HE_result);
    // HE_result = cryptoContext_->Compress(HE_result, 1);

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result, "after mod-switch");
#endif
    send_encrypted_vector_openfhe(io, HE_result);
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
//   if (slot_count > POLY_MOD_DEGREE) {
//     free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_,
//               zero_);
//   }
}

// void BERTFCField::verify(vector<uint64_t> *vec, vector<uint64_t *> *matrix,
//                      vector<vector<uint64_t>> &C) {
//   if (party == BOB) {
//     io->send_data(vec->data(), data.filter_w * sizeof(uint64_t));
//     io->flush();
//     for (int i = 0; i < data.filter_h; i++) {
//       io->send_data(C[i].data(), sizeof(uint64_t));
//     }
//   } else // party == ALICE
//   {
//     vector<uint64_t> vec_0(data.filter_w);
//     io->recv_data(vec_0.data(), data.filter_w * sizeof(uint64_t));
//     for (int i = 0; i < data.filter_w; i++) {
//       vec_0[i] = (vec_0[i] + (*vec)[i]) % prime_mod;
//     }
//     auto result = ideal_functionality(vec_0.data(), matrix->data());

//     vector<vector<uint64_t>> C_0(data.filter_h);
//     for (int i = 0; i < data.filter_h; i++) {
//       C_0[i].resize(1);
//       io->recv_data(C_0[i].data(), sizeof(uint64_t));
//       C_0[i][0] = (C_0[i][0] + C[i][0]) % prime_mod;
//     }
//     bool pass = true;
//     for (int i = 0; i < data.filter_h; i++) {
//       if (neg_mod(result[i], (int64_t)prime_mod) != (int64_t)C_0[i][0]) {
//         pass = false;
//       }
//     }
//     if (pass)
//       cout << GREEN << "[Server] Successful Operation" << RESET << endl;
//     else {
//       cout << RED << "[Server] Failed Operation" << RESET << endl;
//       cout << RED << "WARNING: The implementation assumes that the computation"
//            << endl;
//       cout << "performed locally by the server (on the model and its input "
//               "share)"
//            << endl;
//       cout << "fits in a 64-bit integer. The failed operation could be a result"
//            << endl;
//       cout << "of overflowing the bound." << RESET << endl;
//     }
//   }
// }
