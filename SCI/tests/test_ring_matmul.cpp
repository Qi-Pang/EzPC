/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
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

#include "LinearOT/linear-ot.h"
#include "utils/emp-tool.h"
#include <iostream>
#include <fstream>

using namespace sci;
using namespace std;

int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
LinearOT *prod;

int dim1 = 128;
int dim2 = 64;
int dim3 = 128;
int bwA = 37;
int bwB = 37;
int bwC = 37;
bool signed_B = true;
bool accumulate = true;
bool precomputed_MSBs = false;
MultMode mode = MultMode::None;

uint64_t maskA = (bwA == 64 ? -1 : ((1ULL << bwA) - 1));
uint64_t maskB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));
uint64_t maskC = (bwC == 64 ? -1 : ((1ULL << bwC) - 1));

std::vector<std::vector<uint64_t>> read_data(const std::string& filename) {
    std::ifstream input_file(filename);
    std::vector<std::vector<uint64_t>> data;

    if (!input_file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(input_file, line)) {
        std::vector<uint64_t> row;
        std::istringstream line_stream(line);
        std::string cell;

        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stoll(cell));
        }

        data.push_back(row);
    }

    input_file.close();
    return data;
}

void test_matrix_multiplication(uint64_t *inA, uint64_t *inB,
                                bool signed_arithmetic = true) {
  int dim = (::accumulate ? dim1 * dim3 : dim1 * dim2 * dim3);
  uint64_t *outC = new uint64_t[dim];

  INIT_TIMER;
  START_TIMER;
  uint8_t *msbA = nullptr;
  uint8_t *msbB = nullptr;
  if (precomputed_MSBs) {
    msbA = new uint8_t[dim1 * dim2];
    msbB = new uint8_t[dim2 * dim3];
    prod->aux->MSB(inA, msbA, dim1 * dim2, bwA);
    prod->aux->MSB(inB, msbB, dim2 * dim3, bwB);
  }
  uint64_t num_rounds = iopack->get_rounds();
  uint64_t comm_start = iopack->get_comm();
  prod->matrix_multiplication(dim1, dim2, dim3, inA, inB, outC, bwA, bwB, bwC,
                              signed_arithmetic, signed_B, ::accumulate, mode,
                              msbA, msbB);
  if (precomputed_MSBs) {
    delete[] msbA;
    delete[] msbB;
  }
  uint64_t comm_end = iopack->get_comm();
  cout << "Bytes Sent: " << (comm_end - comm_start) << endl;
  num_rounds = iopack->get_rounds() - num_rounds;
  cout << "Num rounds: " << num_rounds << endl;
  STOP_TIMER("Total time for matmul");

  if (party == ALICE) {
    iopack->io->send_data(inA, dim1 * dim2 * sizeof(uint64_t));
    iopack->io->send_data(inB, dim2 * dim3 * sizeof(uint64_t));
    iopack->io->send_data(outC, dim * sizeof(uint64_t));
  } else { // party == BOB
    uint64_t *inA0 = new uint64_t[dim1 * dim2];
    uint64_t *inB0 = new uint64_t[dim2 * dim3];
    uint64_t *outC0 = new uint64_t[dim];
    iopack->io->recv_data(inA0, dim1 * dim2 * sizeof(uint64_t));
    iopack->io->recv_data(inB0, dim2 * dim3 * sizeof(uint64_t));
    iopack->io->recv_data(outC0, dim * sizeof(uint64_t));

    int extra_bits = (::accumulate ? ceil(log2(dim2)) : 0);
    uint64_t *res = new uint64_t[dim];
    for (int i = 0; i < dim1 * dim2; i++) {
      if (signed_arithmetic) {
        if (mode == MultMode::Alice_has_A) {
          inA0[i] = signed_val(inA0[i], bwA);
        } else if (mode == MultMode::Bob_has_A) {
          inA0[i] = signed_val(inA[i], bwA);
        } else {
          inA0[i] = signed_val(inA0[i] + inA[i], bwA);
        }
      } else {
        if (mode == MultMode::Alice_has_A) {
          inA0[i] = unsigned_val(inA0[i], bwA);
        } else if (mode == MultMode::Bob_has_A) {
          inA0[i] = unsigned_val(inA[i], bwA);
        } else {
          inA0[i] = unsigned_val(inA0[i] + inA[i], bwA);
        }
      }
    }
    for (int i = 0; i < dim2 * dim3; i++) {
      if (signed_arithmetic && signed_B) {
        if (mode == MultMode::Alice_has_B) {
          inB0[i] = signed_val(inB0[i], bwB);
        } else if (mode == MultMode::Bob_has_B) {
          inB0[i] = signed_val(inB[i], bwB);
        } else {
          inB0[i] = signed_val(inB0[i] + inB[i], bwB);
        }
      } else {
        if (mode == MultMode::Alice_has_B) {
          inB0[i] = unsigned_val(inB0[i], bwB);
        } else if (mode == MultMode::Bob_has_B) {
          inB0[i] = unsigned_val(inB[i], bwB);
        } else {
          inB0[i] = unsigned_val(inB0[i] + inB[i], bwB);
        }
      }
    }
    cout << "debug " << signed_val(outC[0] + outC0[0], bwC) << endl;
    for (int i = 0; i < dim2; i++) {
      cout << inA0[i] << " ";
    }
    cout << endl;
    for (int i = 0; i < dim2; i++) {
      cout << inB0[i * dim1] << " ";
    }
    cout << endl;
    prod->matmul_cleartext(dim1, dim2, dim3, inA0, inB0, res, ::accumulate);
    for (int i = 0; i < dim3; i++) {
      auto temp = signed_val(outC[i] + outC0[i], bwC);
      cout << temp << " ";
    }
    cout << endl;

    for (int i = 0; i < dim; i++) {
      if (signed_arithmetic) {
        // assert(signed_val(res[i] >> extra_bits, bwC) ==
        //        signed_val(outC[i] + outC0[i], bwC));
          assert(signed_val(res[i], bwC) ==
               signed_val(outC[i] + outC0[i], bwC));
      } else {
        // assert(unsigned_val(res[i] >> extra_bits, bwC) ==
        //        unsigned_val(outC[i] + outC0[i], bwC));
        assert(unsigned_val(res[i], bwC) ==
               unsigned_val(outC[i] + outC0[i], bwC));
      }
    }
    if (signed_arithmetic)
      cout << "SMult Tests Passed" << endl;
    else
      cout << "UMult Tests Passed" << endl;

    delete[] inA0;
    delete[] inB0;
    delete[] outC0;
    delete[] res;
  }
  delete[] outC;
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("m", precomputed_MSBs, "MSB_to_Wrap Optimization?");
  amap.arg("a", ::accumulate, "Accumulate?");

  amap.parse(argc, argv);

  iopack = new IOPack(party, port, "127.0.0.1");
  otpack = new OTPack(iopack, party);
  prod = new LinearOT(party, iopack, otpack);

  PRG128 prg; //(fix_key);

  uint64_t *inA = new uint64_t[dim1 * dim2];
  uint64_t *inB = new uint64_t[dim2 * dim3];

  prg.random_data(inA, dim1 * dim2 * sizeof(uint64_t));
  prg.random_data(inB, dim2 * dim3 * sizeof(uint64_t));

  vector<vector<uint64_t>> tempA = read_data("./bin/txt/random_ct1.txt");
  vector<vector<uint64_t>> tempB = read_data("./bin/txt/random_ct2.txt");

  for (int i = 0; i < dim1 * dim2; i++) {
    inA[i] &= maskA;
  }
  for (int i = 0; i < dim2 * dim3; i++) {
    inB[i] &= maskB;
  }

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      inA[i * dim2 + j] = tempA[i][j] * 4;
    }
  }
  for (int i = 0; i < dim2; i++) {
    for (int j = 0; j < dim3; j++) {
      inB[i * dim3 + j] = tempB[i][j] * 4;
    }
  }

  cout << "Precomputed MSBs: " << precomputed_MSBs << endl;
  cout << "Accumulate: " << ::accumulate << endl;
  mode = MultMode::None;
  cout << "Mode: None" << endl;
  test_matrix_multiplication(inA, inB, false);
  test_matrix_multiplication(inA, inB, true);
  // mode = MultMode::Alice_has_A;
  // cout << "Mode: Alice_has_A" << endl;
  // test_matrix_multiplication(inA, inB, false);
  // test_matrix_multiplication(inA, inB, true);
  // mode = MultMode::Alice_has_B;
  // cout << "Mode: Alice_has_B" << endl;
  // test_matrix_multiplication(inA, inB, false);
  // test_matrix_multiplication(inA, inB, true);
  // mode = MultMode::Bob_has_A;
  // cout << "Mode: Bob_has_A" << endl;
  // test_matrix_multiplication(inA, inB, false);
  // test_matrix_multiplication(inA, inB, true);
  // mode = MultMode::Bob_has_B;
  // cout << "Mode: Bob_has_B" << endl;
  // test_matrix_multiplication(inA, inB, false);
  // test_matrix_multiplication(inA, inB, true);

  delete[] inA;
  delete[] inB;
  delete prod;
}
