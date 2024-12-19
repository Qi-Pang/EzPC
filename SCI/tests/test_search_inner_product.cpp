/*
Authors: Qi Pang
*/
#include "LinearHE/enc_search.h"
#include <fstream>

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int bitlength = 32; // not used
int num_threads = 16; // not used
int port = 8000;
string address = "127.0.0.1";
int input_dim = 1; // batch size
int common_dim = 4096; // dimension
int output_dim = 16 * 1024; // DB size
int filter_precision = 15; // not used

std::vector<std::vector<uint64_t>> read_data(const std::string& filename, int slot_count) {
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

    #pragma omp parallel for
    for (int i = 0; i < data.size(); i++) {
        data[i].resize(slot_count, 0ULL);
    }
    return data;
}

void Search(SearchCTPT &befc, int32_t input_dim, int32_t common_dim, int32_t output_dim) {
    vector<vector<uint64_t>> A;   // Inputs
    vector<vector<uint64_t>> B1;  // DB embeddings

    PRG128 prg;

    A = read_data("./input_img.txt", befc.slot_count / 2);
    B1 = read_data("./db_embedding.txt", befc.slot_count / 2);

    cout << "A size: " << A.size() << " x " << A[0].size() << endl;
    cout << "B1 size: " << B1.size() << " x " << B1[0].size() << endl;

    cout << "prime: " << prime_mod << endl;
    INIT_TIMER;
    START_TIMER;
    befc.search(input_dim, common_dim, output_dim, A, B1, true);
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

    // 28 bits
    prime_mod = 268582913;

    // 25 bits
    // prime_mod = 33832961;
    
    // 16 bits
    // prime_mod = 65537;

    // 17 bits
    // prime_mod = 147457;
    
    // 18 bits
    // prime_mod = 270337;

    // 19 bits
    // prime_mod = 557057;

    // 20 bits
    // prime_mod = 1073153;

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

    SearchCTPT befc(party, io);
    cout << "Before MatMul" << endl;
    Search(befc, input_dim, common_dim, output_dim);

    cout << "Communication Round: " << io->num_rounds << endl;

    io->flush();
    return 0;
}
