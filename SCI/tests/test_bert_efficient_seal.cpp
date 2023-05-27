/*
Authors: Qi Pang
*/
#include "LinearHE/bert-matmul-cipher-efficient-seal.h"
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
int common_dim = 768;
int output_dim = 64;
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

vector<vector<vector<uint64_t>>> read_qkv_weights(const string& filename) {
    std::ifstream input_file(filename);
    vector<vector<vector<uint64_t>>> data;

    if (!input_file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    vector<vector<uint64_t>> sub_mat;
    int row_counting = 0;
    while (std::getline(input_file, line)) {
        vector<uint64_t> row;
        istringstream line_stream(line);
        string cell;

        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stoll(cell));
        }
        sub_mat.push_back(row);
        row_counting++;
        if (row_counting == 768) {
            data.push_back(sub_mat);
            sub_mat.clear();
            row_counting -= 768;
        }
    }

    input_file.close();
    return data;
}

vector<vector<uint64_t>> read_qkv_bias(const string& filename) {
    std::ifstream input_file(filename);
    vector<vector<uint64_t>> data;

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
        if (row_counting == 64) {
            data.push_back(sub_mat);
            sub_mat.clear();
            row_counting -= 64;
        }
    }

    input_file.close();
    return data;
}

void MatMul(BEFCField &befc, int32_t input_dim, int32_t common_dim, int32_t output_dim) {
    vector<vector<uint64_t>> A(input_dim);   // Inputs
    vector<vector<uint64_t>> B1(common_dim);  // Weights
    vector<vector<uint64_t>> B2(common_dim);  // Weights
    vector<vector<uint64_t>> B3(common_dim);  // Weights
    vector<uint64_t> Bias1(output_dim);  // Weights
    vector<uint64_t> Bias2(output_dim);  // Weights
    vector<uint64_t> Bias3(output_dim);  // Weights
    vector<vector<uint64_t>> C(input_dim);   // Outputs
    PRG128 prg;
    for (int i = 0; i < common_dim; i++) {
        B1[i].resize(output_dim);
        B2[i].resize(output_dim);
        // C[i].resize(output_dim);
        if (party == ALICE) {  // Server
            prg.random_data(B1[i].data(), output_dim * sizeof(uint64_t));
            prg.random_data(B2[i].data(), output_dim * sizeof(uint64_t));
            for (int j = 0; j < output_dim; j++) {
                B1[i][j] = ((int64_t)B1[i][j]) >> (64 - filter_precision);
                B2[i][j] = ((int64_t)B2[i][j]) >> (64 - filter_precision);
            }
        }
    }
    for (int i = 0; i < input_dim; i++) {
        A[i].resize(common_dim);
        C[i].resize(output_dim);
    }

    for(int i = 0; i < output_dim; i++) {
        Bias1[i] = i;
        Bias2[i] = 64 - i;
        Bias3[i] = i;
    }

    // A = read_data("./bin/txt/X_quantize_0.txt");
    // B1 = read_data("./bin/txt/Q_quantize_0.txt");
    // B2 = read_data("./bin/txt/K_quantize_0.txt");

    A = read_data("./bin/txt/random_X.txt");
    B1 = read_data("./bin/txt/random_Y.txt");
    B2 = read_data("./bin/txt/random_Z.txt");
    B3 = read_data("./bin/txt/random_Z.txt");

    // auto temp_w = read_qkv_weights("/home/qipang/mnt/d2/secure-bert/robert/sparse/sst-2/weights_txt/bert.encoder.layer.0.attention.self.query.weight.txt");
    // auto temp_b = read_qkv_bias("/home/qipang/mnt/d2/secure-bert/robert/sparse/sst-2/weights_txt/bert.encoder.layer.0.attention.self.query.bias.txt");

    // cout << temp_w.size() << " " << temp_w[0].size() << " " << temp_w[0][1].size() << endl;;
    // cout << temp_b.size() << " " << temp_b[0].size() << endl;

    cout << "prime: " << prime_mod << endl;
    INIT_TIMER;
    START_TIMER;
    befc.matrix_multiplication(input_dim, common_dim, output_dim, A, B1, B2, B3, Bias1, Bias2, Bias3, C, true);
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
    prime_mod = 536903681;

    // 28bits
    // prime_mod = 268582913;

    // 25 bits
    // prime_mod = 33832961;
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
