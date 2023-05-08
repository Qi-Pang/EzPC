/*
Authors: Qi Pang
*/
#include "LinearHE/bert-matmul-cipher-efficient-seal.h"

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int bitlength = 32;
int num_threads = 16;
int port = 8000;
string address = "127.0.0.1";
int input_dim = 128;
int common_dim = 768;
int output_dim = 64;
int filter_precision = 15;

void MatMul(BEFCField &befc, int32_t input_dim, int32_t common_dim, int32_t output_dim) {
//   int num_cols = 1;
  vector<vector<uint64_t>> A(input_dim);   // Inputs
  vector<vector<uint64_t>> B(common_dim);  // Weights
  vector<vector<uint64_t>> C(input_dim);   // Outputs
  PRG128 prg;
  for (int i = 0; i < common_dim; i++) {
    B[i].resize(output_dim);
    // C[i].resize(output_dim);
    if (party == ALICE) {  // Server
      prg.random_data(B[i].data(), output_dim * sizeof(uint64_t));
      for (int j = 0; j < output_dim; j++) {
        B[i][j] = ((int64_t)B[i][j]) >> (64 - filter_precision);
      }
    }
  }
  for (int i = 0; i < input_dim; i++) {
    A[i].resize(common_dim);
    C[i].resize(output_dim);
    // prg.random_mod_p<uint64_t>(A[i].data(), common_dim, prime_mod);
  }

  for (int i = 0; i < input_dim; i++)
    for (int j = 0; j < common_dim; j++)
        A[i][j] = i + j;
        // A[i][j] = 1;

  for (int i = 0; i < common_dim; i++)
    for (int j = 0; j < output_dim; j++)
        B[i][j] = i + j;
        // B[i][j] = 1;

  cout << "prime: " << prime_mod << endl;
  INIT_TIMER;
  START_TIMER;
  befc.matrix_multiplication(input_dim, common_dim, output_dim, A, B, C, true,
                              true);
  STOP_TIMER("Total Time for FC");
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("n", input_dim, "Input Dim");
  amap.arg("c", common_dim, "Common Dim");
  amap.arg("k", output_dim, "Output Dim");
  amap.arg("fp", filter_precision, "Filter Precision");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("l", bitlength, "Bitlength of inputs");
  amap.parse(argc, argv);
  // prime_mod = default_prime_mod.at(bitlength);
  // prime_mod = 268582913;
  prime_mod = 33832961;
  // prime_mod = 65537;

  cout << "===================================================================="
       << endl;
  cout << "Role: " << party << " - Bitlength: " << bitlength
       << " - Mod: " << prime_mod << " - InputDim: " << input_dim
       << " - CommonDim: " << common_dim << " - OutputDim: " << output_dim << 
       " - # Threads: " << num_threads << endl;
  cout << "===================================================================="
       << endl;

  NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

  auto io_start = io->counter;

  BEFCField befc(party, io);
  cout << "Before MatMul" << endl;
  MatMul(befc, input_dim, common_dim, output_dim);

  cout << "Communication Round: " << io->num_rounds << endl;

  io->flush();
  return 0;
}
