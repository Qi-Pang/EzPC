/*
Authors: Qi Pang
*/
#include "LinearHE/iron-att-linear.h"
#include <fstream>

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int bitlength = 32;
int num_threads = 12;
int port = 8000;
string address = "127.0.0.1";
int input_dim = 128;
int common_dim = 768;
int output_dim = 64;
int filter_precision = 15;

#define MAX_THREADS 12

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

void MatMul(IRONFC &befc, int32_t input_dim, int32_t common_dim, int32_t output_dim) {
    vector<vector<uint64_t>> A(input_dim);   // Inputs
    vector<vector<uint64_t>> B1(common_dim);  // Weights
    vector<vector<uint64_t>> B2(common_dim);  // Weights
    vector<vector<uint64_t>> Bias(12, vector<uint64_t>(output_dim, 0ULL));  // Weights
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

    // A = read_data("./bin/txt/X_quantize_0.txt");
    // B1 = read_data("./bin/txt/Q_quantize_0.txt");
    // B2 = read_data("./bin/txt/K_quantize_0.txt");

    // A = read_data("./bin/txt/random_X.txt");
    // B1 = read_data("./bin/txt/random_Y.txt");
    // B2 = read_data("./bin/txt/random_Z.txt");

    A = read_data("/home/qipang/mnt/d2/iron/mrpc/weights_txt_right/inputs_0_data.txt");
    // auto temp_w1 = read_qkv_weights("/home/qipang/mnt/d2/secure-bert/robert/sparse/sst-2/weights_txt/bert.encoder.layer.0.attention.self.query.weight.txt");
    // auto temp_w2 = read_qkv_weights("/home/qipang/mnt/d2/secure-bert/robert/sparse/sst-2/weights_txt/bert.encoder.layer.0.attention.self.key.weight.txt");

    auto temp_w1 = read_qkv_weights("/home/qipang/mnt/d2/iron/mrpc/weights_txt_right/bert.encoder.layer.0.attention.self.query.weight.txt");
    auto temp_w2 = read_qkv_weights("/home/qipang/mnt/d2/iron/mrpc/weights_txt_right/bert.encoder.layer.0.attention.self.key.weight.txt");
    
    auto temp_w3 = read_qkv_weights("/home/qipang/mnt/d2/iron/mrpc/weights_txt_right/bert.encoder.layer.0.attention.self.value.weight.txt");
    auto temp_b1 = read_qkv_bias("/home/qipang/mnt/d2/iron/mrpc/weights_txt_right/bert.encoder.layer.0.attention.self.query.bias.txt");
    auto temp_b2 = read_qkv_bias("/home/qipang/mnt/d2/iron/mrpc/weights_txt_right/bert.encoder.layer.0.attention.self.key.bias.txt");
    auto temp_b3 = read_qkv_bias("/home/qipang/mnt/d2/iron/mrpc/weights_txt_right/bert.encoder.layer.0.attention.self.value.bias.txt");


    cout << "prime: " << prime_mod << endl;
    INIT_TIMER;
    START_TIMER;
    befc.matrix_multiplication(input_dim, common_dim, output_dim, A, temp_w1, temp_w2, temp_w3, temp_b1, temp_b2, temp_b3, C, true);
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
    prime_mod = (uint64_t) pow(2, 37);

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


    // uint64_t thread_comm[num_threads + 1];
    uint64_t total_comm = 0;

    NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port);
    // IOPack *iopackArr[MAX_THREADS];
    // OTPack *otpackArr[MAX_THREADS];

    // for (int i = 0; i < num_threads; i++) {
    //     iopackArr[i] = new IOPack(party, port + i, address.c_str());
    //     otpackArr[i] = new OTPack(iopackArr[i], party);
    // }

    // // IOPack *iopack = new IOPack(party, port, address.c_str());
    // NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port+num_threads);

    auto io_start = io->counter;

    // for (int i = 0; i < num_threads; i++) {
    //     thread_comm[i] = iopackArr[i]->get_comm();
    // }
    // thread_comm[num_threads] = io_start;

    IRONFC befc(party, io);
    MatMul(befc, input_dim, common_dim, output_dim);

    // for (int i = 0; i < num_threads; i++) {
    //     thread_comm[i] = iopackArr[i]->get_comm() - thread_comm[i];
    //     total_comm += thread_comm[i];
    // }
    // thread_comm[num_threads] = io->counter - io_start;
    // total_comm += thread_comm[num_threads];
    total_comm  = io->counter - io_start;
    cout << "Communication Sent\t" << total_comm << " bytes" << endl;
    cout << "Communication Round: " << io->num_rounds << endl;

    io->flush();
    return 0;
}
