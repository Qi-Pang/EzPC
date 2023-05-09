/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "LinearHE/utils-HE.h"

using namespace std;
using namespace sci;
using namespace seal;
using namespace seal::util;

// TODO: slow, fix this
void send_encrypted_vector(NetIO *io, vector<Ciphertext> &ct_vec) {
  assert(ct_vec.size() > 0);
  for (size_t ct = 0; ct < ct_vec.size(); ct++) {
    stringstream os;
    uint64_t ct_size;
    ct_vec[ct].save(os);
    ct_size = os.tellp();
    string ct_ser = os.str();
    io->send_data(&ct_size, sizeof(uint64_t));
    io->send_data(ct_ser.c_str(), ct_ser.size());
    }
}

void recv_encrypted_vector(SEALContext* context_, NetIO *io, vector<Ciphertext> &ct_vec) {
  assert(ct_vec.size() > 0);
  for (size_t ct = 0; ct < ct_vec.size(); ct++) {
    stringstream is;
    uint64_t ct_size;
    io->recv_data(&ct_size, sizeof(uint64_t));
    char *c_enc_result = new char[ct_size];
    io->recv_data(c_enc_result, ct_size);
    is.write(c_enc_result, ct_size);
    ct_vec[ct].unsafe_load(*context_, is);
    delete[] c_enc_result;
  }
}

void send_ciphertext(NetIO *io, Ciphertext &ct) {
  stringstream os;
  uint64_t ct_size;
  ct.save(os);
  ct_size = os.tellp();
  string ct_ser = os.str();
  io->send_data(&ct_size, sizeof(uint64_t));
  io->send_data(ct_ser.c_str(), ct_ser.size());
}

void recv_ciphertext(SEALContext* context_, NetIO *io, Ciphertext &ct) {
  stringstream is;
  uint64_t ct_size;
  io->recv_data(&ct_size, sizeof(uint64_t));
  char *c_enc_result = new char[ct_size];
  io->recv_data(c_enc_result, ct_size);
  is.write(c_enc_result, ct_size);
  ct.unsafe_load(*context_, is);
  delete[] c_enc_result;
}
