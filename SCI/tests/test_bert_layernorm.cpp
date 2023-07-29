/*
Authors: Qi Pang
*/
#include "LinearHE/bert-layernorm.h"
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

void LayerNorm(LayerNormField &beln, int32_t input_dim, int32_t common_dim, int32_t output_dim) {
    vector<vector<uint64_t>> 

    cout << "prime: " << prime_mod << endl;
    INIT_TIMER;
    START_TIMER;

    vector<vector<uint64_t>> X1(input_dim, vector<unint64_t>(common_dim, 0ULL));
    vector<vector<uint64_t>> X2(input_dim, vector<unint64_t>(common_dim, 0ULL));
    vector<uint64_t> Gamma(common_dim, 0ULL);
    vector<uint64_t> Var1(input_dim, 0ULL);
    vector<uint64_t> Var2(input_dim, 0ULL);

    beln.layernorm_he(input_dim, common_dim, output_dim, X1, X2, Gamma, Var1, Var2);
    STOP_TIMER("Total Time for LN");
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

    prime_mod = std::pow(2, 37);

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

    LayerNormField beln(party, io);
    cout << "Before MatMul" << endl;
    LayerNorm(beln, input_dim, common_dim, output_dim);

    cout << "Communication Round: " << io->num_rounds << endl;

    io->flush();
    return 0;
}
