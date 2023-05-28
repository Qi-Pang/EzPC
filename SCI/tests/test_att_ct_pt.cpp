/*
Authors: Qi Pang
*/
#include "LinearHE/bert-att-ct-pt.h"
#include <fstream>

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int bitlength = 32;
int num_threads = 16;
int port = 8000;
string address = "127.0.0.1";
int input_dim = 128;
int common_dim = 3072;
int output_dim = 768;
int filter_precision = 15;

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

vector<uint64_t> read_bias(const string& filename, int output_dim) {
    std::ifstream input_file(filename);
    vector<uint64_t> data;

    if (!input_file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    vector<uint64_t> sub_mat;
    int row_counting = 0;
    while (std::getline(input_file, line)) {
        istringstream line_stream(line);
        string cell;

        while (std::getline(line_stream, cell, ',')) {
            sub_mat.push_back(std::stoll(cell));
        }
        row_counting++;
        if (row_counting == output_dim) {
            data = sub_mat;
            sub_mat.clear();
            row_counting -= output_dim;
        }
    }

    input_file.close();
    return data;
}

void MatMul(BEAttCtPt &befc, int32_t input_dim, int32_t common_dim, int32_t output_dim) {
    vector<vector<uint64_t>> A(input_dim);   // Inputs
    vector<vector<uint64_t>> B(common_dim);  // Weights
    vector<vector<uint64_t>> C(input_dim);   // Outputs
    vector<uint64_t> Bias(output_dim);   // Bias
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
    }

    // A = read_data("./bin/txt/X_quantize_0.txt");
    // B = read_data("./bin/txt/Q_quantize_0.txt");
    assert (input_dim == 128);
    if (common_dim == 3072 && output_dim == 768)
    {
        A = read_data("./bin/txt/random_X_inter1.txt");
        B = read_data("./bin/txt/random_Y_inter1.txt");
        Bias = read_bias("./bin/txt/bias_768.txt", output_dim);
    }
    else if (common_dim == 768 && output_dim == 3072)
    {
        A = read_data("./bin/txt/random_X_attout.txt");
        B = read_data("./bin/txt/random_Y_inter2.txt");
        Bias = read_bias("./bin/txt/bias_3072.txt", output_dim);
    }
    else if (common_dim == 768 && output_dim == 768) {
        A = read_data("./bin/txt/random_X_attout.txt");
        B = read_data("./bin/txt/random_Y_attout.txt");
        Bias = read_bias("./bin/txt/bias_768.txt", output_dim);
    }
    else {
        assert (0);
    }

    cout << "prime: " << prime_mod << endl;
    INIT_TIMER;
    START_TIMER;
    befc.matrix_multiplication(input_dim, common_dim, output_dim, A, B, Bias, C, true);
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

    // 32 bits
    // prime_mod = 4293918721;

    // 30 bits
    // prime_mod = 1073872897; 

    // 29 bits 
    // prime_mod = 536903681;

    // 37 bits
    // prime_mod = 137439010817;

    // 17 bits
    // prime_mod = 147457;

    // 19 bits
    prime_mod = 557057;

    // 28 bits
    // prime_mod = 268582913;

    // 25 bits
    // prime_mod = 33832961;
    // prime_mod = 65537;

    cout << "===================================================================="
        << endl;
    cout << "Role: " << party << " - Mod: " << prime_mod << " - InputDim: " << input_dim
        << " - CommonDim: " << common_dim << " - OutputDim: " << output_dim << endl;
    cout << "===================================================================="
        << endl;

    NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

    auto io_start = io->counter;

    BEAttCtPt befc(party, io);
    MatMul(befc, input_dim, common_dim, output_dim);

    cout << "Communication Round: " << io->num_rounds << endl;

    io->flush();
    return 0;
}
