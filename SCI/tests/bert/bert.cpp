#include "bert.h"

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

Bert::Bert(int party, int port, string address){
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

    cout << "> Setup Linear" << endl;
    this->lin = Linear(party, io);
    cout << "> Setup NonLinear" << endl;
    this->nl = NonLinear(party, address, port+1);
    cout << "> Bert intialized done!" << endl << endl;

    data_lin1.filter_h = COMMON_DIM;
    data_lin1.filter_w = OUTPUT_DIM;
    data_lin1.image_size = INPUT_DIM;
    data_lin1.slot_count = 8192;

}

Bert::~Bert() {
    
}

void Bert::run_server() {
    cout << "> Loading weights and bias" << endl;
    // Loading weights
    // Weights: 768 x 64
    vector<vector<vector<uint64_t>>> wq = read_qkv_weights(
        "./weights_txt/bert.encoder.layer.0.attention.self.query.weight.txt");
    vector<vector<vector<uint64_t>>> wk = read_qkv_weights(
        "./weights_txt/bert.encoder.layer.0.attention.self.key.weight.txt");
    vector<vector<vector<uint64_t>>> wv = read_qkv_weights(
        "./weights_txt/bert.encoder.layer.0.attention.self.value.weight.txt");

    // // Bias: 768 x 64
    vector<vector<uint64_t>> bq = read_qkv_bias(
        "./weights_txt/bert.encoder.layer.0.attention.self.query.bias.txt");
    vector<vector<uint64_t>> bk = read_qkv_bias(
        "./weights_txt/bert.encoder.layer.0.attention.self.key.bias.txt");
    vector<vector<uint64_t>> bv = read_qkv_bias(
        "./weights_txt/bert.encoder.layer.0.attention.self.value.bias.txt");

    vector<vector<vector<vector<uint64_t>>>> w_q;
    vector<vector<vector<vector<uint64_t>>>> w_k;
    vector<vector<vector<vector<uint64_t>>>> w_v;
    vector<vector<vector<uint64_t>>> w_o;
    vector<vector<vector<uint64_t>>> w_i_1;
    vector<vector<vector<uint64_t>>> w_i_2;

    vector<vector<vector<uint64_t>>> b_q;
    vector<vector<vector<uint64_t>>> b_k;
    vector<vector<vector<uint64_t>>> b_v;
    vector<vector<vector<uint64_t>>> b_o;
    vector<vector<vector<uint64_t>>> b_i_1;
    vector<vector<vector<uint64_t>>> b_i_2;

    // Temporal 
    w_q.push_back(wq);
    w_k.push_back(wk);
    w_v.push_back(wv);

    b_q.push_back(bq);
    b_k.push_back(bk);
    b_v.push_back(bv);

    cout << "> Entering Attention Layers" << endl;
    for(int layer_id; layer_id < ATTENTION_LAYERS; ++layer_id){
        cout << "-> Layer - " << layer_id << endl;
        // Receive cipher text input
        vector<Ciphertext> h1(12);
        recv_encrypted_vector(lin.he_8192->context, io, h1);

        cout << "-> Layer - " << layer_id << ": Receive input cts from client " << endl;

        // -------------------- Linear #1 -------------------- //
        vector<Ciphertext> l1_result = lin.linear_1(
            lin.he_8192,
            h1,
            w_q[layer_id],
            w_k[layer_id],
            w_v[layer_id],
            b_q[layer_id],
            b_k[layer_id],
            b_v[layer_id],
            data_lin1
        );

        cout << "-> Layer - " << layer_id << ": Linear #1 done " << endl;

        continue;

        // -------------------- Softmax -------------------- //
        // WARING: result from he is column packing
        uint64_t softmax_input_col_pack[ATTENTION_LAYERS*INPUT_DIM*INPUT_DIM];
        uint64_t softmax_input[ATTENTION_LAYERS*INPUT_DIM*INPUT_DIM];
        uint64_t softmax_output[ATTENTION_LAYERS*INPUT_DIM*INPUT_DIM];
        uint64_t softmax_output_col_pack[ATTENTION_LAYERS*INPUT_DIM*INPUT_DIM];

        // he_to_ss_server(lin.he_8192, l1_result, softmax_input_col_pack);

        // // TODO: reorganize layout
        // // Get softmax_input

        // nl.softmax(
        //     NL_NTHREADS,
        //     softmax_input,
        //     softmax_output,
        //     ATTENTION_LAYERS*INPUT_DIM,
        //     INPUT_DIM,
        //     NL_ELL,
        //     NL_SCALE);

        // // TODO:reorganize layout

        // vector<Ciphertext> h2 = ss_to_he_server(
        //     lin.he_4096, 
        //     softmax_output_col_pack);

        // // -------------------- Linear #2 -------------------- //
        // vector<Ciphertext> h2(24);
        // vector<Ciphertext> h3 = lin.linear_2(
        //     h2, 
        //     w_o[layer_id],
        //     b_o[layer_id],
        //     data_lin2
        // );

        // // -------------------- Layer Norm -------------------- //

        // // ------------------ Linear inter #1 ------------------ //
        // vector<Ciphertext> h4(24);
        // vector<Ciphertext> h5 = lin.linear_inter(
        //     h4, 
        //     w_i_1[layer_id],
        //     b_i_1[layer_id],
        //     data_lin3
        // );

        // // ---------------------- GELU ---------------------- //

        // // ------------------ Linear inter #2 ------------------ //
        // vector<Ciphertext> h6(24);
        // vector<Ciphertext> h7 = lin.linear_inter(
        //     h6, 
        //     w_i_2[layer_id],
        //     b_i_2[layer_id],
        //     data_lin4
        // );

        // // -------------------- Layer Norm -------------------- //


    }

    // -------------------- POOL -------------------- //

    // -------------------- TANH -------------------- //

}

void Bert::run_client() {
    cout << "> Loading input" << endl;
    // Loading inputs 
    // H_1: 128×768
    vector<vector<uint64_t>> h1 = read_data("./txt/random_X.txt");

    // Encrypt input in HE and send to server

    // Column Packing
    vector<uint64_t> h1_vec(COMMON_DIM * INPUT_DIM);
        for (int j = 0; j < COMMON_DIM; j++)
            for (int i = 0; i < INPUT_DIM; i++)
                h1_vec[j*INPUT_DIM + i] = h1[i][j];

    vector<Ciphertext> h1_cts = 
        lin.bert_efficient_preprocess_vec(lin.he_8192, h1_vec, data_lin1);
    
    cout << "> Sending input cts to server" << endl;
    send_encrypted_vector(io, h1_cts);

    for(int layer_id; layer_id < ATTENTION_LAYERS; ++layer_id){

    }

}