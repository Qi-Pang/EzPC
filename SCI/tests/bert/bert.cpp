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

    data_lin2.filter_h = COMMON_DIM;
    data_lin2.filter_w = COMMON_DIM;
    data_lin2.image_size = INPUT_DIM;
    data_lin2.slot_count = 8192;

    data_lin3.filter_h = COMMON_DIM;
    data_lin3.filter_w = INTER_DIM;
    data_lin3.image_size = INPUT_DIM;
    data_lin3.slot_count = 8192;

    data_lin4.filter_h = INTER_DIM;
    data_lin4.filter_w = COMMON_DIM;
    data_lin4.image_size = INPUT_DIM;
    data_lin4.slot_count = 8192;

}

Bert::~Bert() {
    
}

void Bert::he_to_ss_server(HE* he, vector<Ciphertext> in, uint64_t* output){
    PRG128 prg;
    int dim = in.size();
    int slot_count = he->poly_modulus_degree;
	prg.random_mod_p<uint64_t>(output, dim*slot_count, he->plain_mod);

    vector<Ciphertext> cts;
    for(int i = 0; i < dim; i++){
        vector<uint64_t> tmp(slot_count);
        for(int i = 0; i < slot_count; ++i){
            tmp[i] = output[dim*slot_count + i];
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        Ciphertext ct; 
        he->evaluator->sub_plain(in[i], pt, ct);
        cts.push_back(ct);
    }
    send_encrypted_vector(io, cts);
}

vector<Ciphertext> Bert::ss_to_he_server(HE* he, uint64_t* input, int length){
    int slot_count = he->poly_modulus_degree;
    uint64_t plain_mod = he->plain_mod;
    vector<Plaintext> share_server;
    int dim = length / slot_count;
    for(int i = 0; i < dim; i++){
        vector<uint64_t> tmp(slot_count);
        for(int i = 0; i < slot_count; ++i){
            tmp[i] = input[dim*slot_count + i] % plain_mod;
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        share_server.push_back(pt);
    }

    vector<Ciphertext> share_client(dim);
    recv_encrypted_vector(he->context, io, share_client);
    for(int i = 0; i < dim; i++){
        he->evaluator->add_plain_inplace(share_client[i], share_server[i]);
    }
    return share_client;
}

void Bert::he_to_ss_client(HE* he, uint64_t* output, int length, const FCMetadata &data){
    vector<Ciphertext> cts(length);
    recv_encrypted_vector(he->context, io, cts);
    for(int i = 0; i < length; i++){
        vector<uint64_t> plain(data.slot_count, 0ULL);
        Plaintext tmp;
        he->decryptor->decrypt(cts[i], tmp);
        he->encoder->decode(tmp, plain);
        std::copy(plain.begin(), plain.end(), &output[i*data.slot_count]);
    }
}

void Bert::ss_to_he_client(HE* he, uint64_t* input, int length){
    int slot_count = he->poly_modulus_degree;
    uint64_t plain_mod = he->plain_mod;
    vector<Ciphertext> cts;
    int dim = length / slot_count;
    for(int i = 0; i < dim; i++){
        vector<uint64_t> tmp(slot_count);
        for(int i = 0; i < slot_count; ++i){
            tmp[i] = input[dim*slot_count + i] % plain_mod;
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        Ciphertext ct; 
        he->encryptor->encrypt(pt, ct);
        cts.push_back(ct);
    }
    send_encrypted_vector(io, cts);
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
    
    vector<vector<uint64_t>> wo = read_data(
        "./weights_txt/bert.encoder.layer.0.attention.output.dense.weight.txt");
    vector<vector<uint64_t>> wi1 = read_data(
        "./weights_txt/bert.encoder.layer.0.intermediate.dense.weight.txt");
    vector<vector<uint64_t>> wi2 = read_data(
        "./weights_txt/bert.encoder.layer.0.output.dense.weight.txt");
    

    // // Bias: 768 x 64
    vector<vector<uint64_t>> bq = read_qkv_bias(
        "./weights_txt/bert.encoder.layer.0.attention.self.query.bias.txt");
    vector<vector<uint64_t>> bk = read_qkv_bias(
        "./weights_txt/bert.encoder.layer.0.attention.self.key.bias.txt");
    vector<vector<uint64_t>> bv = read_qkv_bias(
        "./weights_txt/bert.encoder.layer.0.attention.self.value.bias.txt");
    
    vector<uint64_t> bo = read_bias(
        "./weights_txt/bert.encoder.layer.0.attention.output.dense.bias.txt", 768);
    vector<uint64_t> bi1 = read_bias(
        "./weights_txt/bert.encoder.layer.0.intermediate.dense.bias.txt", 3072);
    vector<uint64_t> bi2 = read_bias(
        "./weights_txt/bert.encoder.layer.0.output.dense.bias.txt", 768);

    vector<vector<vector<vector<uint64_t>>>> w_q;
    vector<vector<vector<vector<uint64_t>>>> w_k;
    vector<vector<vector<vector<uint64_t>>>> w_v;
    vector<vector<vector<uint64_t>>> w_o;
    vector<vector<vector<uint64_t>>> w_i_1;
    vector<vector<vector<uint64_t>>> w_i_2;

    vector<vector<vector<uint64_t>>> b_q;
    vector<vector<vector<uint64_t>>> b_k;
    vector<vector<vector<uint64_t>>> b_v;
    vector<vector<uint64_t>> b_o;
    vector<vector<uint64_t>> b_i_1;
    vector<vector<uint64_t>> b_i_2;

    // Temporal 
    w_q.push_back(wq);
    w_k.push_back(wk);
    w_v.push_back(wv);
    w_o.push_back(wo);
    w_i_1.push_back(wi1);
    w_i_2.push_back(wi2);
    

    b_q.push_back(bq);
    b_k.push_back(bk);
    b_v.push_back(bv);
    b_o.push_back(bo);
    b_i_1.push_back(bi1);
    b_i_2.push_back(bi2);

    // Receive cipher text input
    vector<Ciphertext> h1(12);
    uint64_t h1_cache[INPUT_DIM*COMMON_DIM] = {0};
    uint64_t h4_cache[INPUT_DIM*COMMON_DIM] = {0};

    recv_encrypted_vector(lin.he_8192->context, io, h1);
    cout << "> Receive input cts from client " << endl;

    cout << "> --- Entering Attention Layers ---" << endl;
    for(int layer_id; layer_id < ATTENTION_LAYERS; ++layer_id){
        cout << "-> Layer - " << layer_id << ": Linear #1 " << endl;

        // -------------------- Linear #1 -------------------- //
        // q_k_v include the result of QxK^T and V
        vector<Ciphertext> q_k_v = lin.linear_1(
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

        // To Secret Share and Post Processing

        int qk_size = PACKING_NUM*INPUT_DIM*INPUT_DIM;
        int v_size = PACKING_NUM*INPUT_DIM*OUTPUT_DIM;
        int softmax_size = PACKING_NUM*INPUT_DIM*INPUT_DIM;
        int att_size = PACKING_NUM*INPUT_DIM*OUTPUT_DIM;
        
        int qk_v_size = qk_size + v_size;

        assert( qk_v_size == q_k_v.size()*(lin.he_8192->poly_modulus_degree));

        uint64_t* qk_v_col_pack = new uint64_t[qk_v_size];
        uint64_t* v_matrix = new uint64_t[v_size];
        uint64_t* softmax_input = new uint64_t[qk_size];
        uint64_t* softmax_output = new uint64_t[softmax_size];
        uint64_t* softmax_v = new uint64_t[att_size];
            
        // Secret sharing and send share to client
        cout << "-> Layer - " << layer_id << ": Secret sharing " << endl;
        he_to_ss_server(lin.he_8192, q_k_v, qk_v_col_pack);
        
        cout << "-> Layer - " << layer_id 
            << ": Softmax preprocessing..." << endl;

        // mod p
        nl.gt_p_sub(
            NL_NTHREADS,
            qk_v_col_pack,
            lin.he_8192->plain_mod,
            qk_v_col_pack,
            qk_v_size,
            NL_ELL,
            NL_SCALE
        );

        lin.plain_cross_packing_postprocess(
            qk_v_col_pack, 
            softmax_input,
            data_lin1);
        
        lin.plain_cross_packing_postprocess_v(
            &qk_v_col_pack[qk_size], 
            v_matrix,
            data_lin1);

        // -------------------- Softmax -------------------- //

        cout << "-> Layer - " << layer_id 
            << ": Softmax and multiply V..." << endl;
        // Softmax
        nl.softmax(
            NL_NTHREADS,
            softmax_input,
            softmax_output,
            12*INPUT_DIM,
            INPUT_DIM,
            NL_ELL,
            NL_SCALE);

        nl.n_matrix_mul(
            NL_NTHREADS,
            softmax_input,
            v_matrix,
            softmax_v,
            PACKING_NUM,
            INPUT_DIM,
            INPUT_DIM,
            OUTPUT_DIM,
            NL_ELL,
            NL_SCALE
        );

        cout << "-> Layer - " << layer_id 
            << ": Softmax postprocessing..." << endl;
        
        vector<vector<uint64_t>> h2_vec = lin.concat(softmax_v, 12, 128, 64);

        uint64_t* h2_col_packing = new uint64_t[INPUT_DIM*COMMON_DIM];
        // Packing before send back to server
        lin.plain_col_packing_preprocess_vec(
            h2_vec,
            h2_col_packing,
            INPUT_DIM,
            COMMON_DIM
        );

        vector<Ciphertext> h2 = ss_to_he_server(
            lin.he_8192, 
            h2_col_packing,
            att_size);

        // Clean up
        delete [] qk_v_col_pack;
        delete [] softmax_input;
        delete [] v_matrix;
        delete [] softmax_output;
        delete [] softmax_v;
        delete [] h2_col_packing;

        // -------------------- Linear #2 -------------------- //

        cout << "-> Layer - " << layer_id << ": Linear #2 " << endl;

        vector<Ciphertext> h3 = lin.linear_2(
            lin.he_8192,
            INPUT_DIM,
            COMMON_DIM,
            COMMON_DIM,
            h2, 
            w_o[layer_id],
            b_o[layer_id],
            data_lin2
        );

        cout << "-> Layer - " << layer_id << ": Linear #2 done " << endl;
        
        // Secret Share

        int ln_size = INPUT_DIM*COMMON_DIM;
        uint64_t* ln_input_col_pack = new uint64_t[ln_size];
        uint64_t* ln_input = new uint64_t[ln_size];
        uint64_t* ln_output = new uint64_t[ln_size];
        uint64_t* ln_output_col_pack = new uint64_t[ln_size];

        // Secret sharing and send share to client
        cout << "-> Layer - " << layer_id << ": Secret Sharing" << endl;
        he_to_ss_server(lin.he_8192, h3, ln_input_col_pack);

        cout << "-> Layer - " << layer_id 
            << ": Layer Norm preprocessing..." << endl;
        // Post Processing
        lin.plain_col_packing_postprocess(
            ln_input_col_pack,
            ln_input,
            data_lin2
        );

        // -------------------- Layer Norm -------------------- //

        // H3 = Linear#2 + H1
        for(int i = 0; i < ln_size; i++){
            ln_input[i] += h1_cache[i];
        }

        cout << "-> Layer - " << layer_id 
            << ": Layer Norm..." << endl;
        nl.layer_norm(
            NL_NTHREADS,
            ln_input,
            ln_output,
            INPUT_DIM,
            COMMON_DIM,
            NL_ELL,
            NL_SCALE
        );

        // update H4
        memcpy(h4_cache, ln_output, ln_size*sizeof(uint64_t));

        cout << "-> Layer - " << layer_id 
            << ": Layer Norm postprocessing..." << endl;
        lin.plain_col_packing_preprocess(
            ln_output,
            ln_output_col_pack,
            INPUT_DIM,
            COMMON_DIM
        );

        vector<Ciphertext> h4 = ss_to_he_server(
            lin.he_8192, 
            ln_output_col_pack,
            INPUT_DIM*COMMON_DIM);

        delete[] ln_input_col_pack;
        delete[] ln_input;
        delete[] ln_output;
        delete[] ln_output_col_pack;

        // ------------------ Linear inter #1 ------------------ //

        cout << "-> Layer - " << layer_id << ": Linear #3 " << endl;
        vector<Ciphertext> h5 = lin.linear_2(
            lin.he_8192,
            INPUT_DIM,
            COMMON_DIM,
            3072,
            h4, 
            w_i_1[layer_id],
            b_i_1[layer_id],
            data_lin3
        );
        cout << "-> Layer - " << layer_id << ": Linear #3 done " << endl;

        int gelu_input_size = 128*3072;
        uint64_t* gelu_input_col_pack =
            new uint64_t[gelu_input_size];
        uint64_t* gelu_input =
            new uint64_t[gelu_input_size];
        uint64_t* gelu_output =
            new uint64_t[gelu_input_size];
        uint64_t* gelu_output_col_pack =
            new uint64_t[gelu_input_size];

        // Secret sharing and send share to client
        he_to_ss_server(lin.he_8192, h5, gelu_input_col_pack);
        cout << "-> Layer - " << layer_id 
            << ": GELU preprocessing..." << endl;

        // Post Processing
        lin.plain_col_packing_postprocess(
            gelu_input_col_pack,
            gelu_input,
            data_lin3
        );

        // ---------------------- GELU ---------------------- //

        cout << "-> Layer - " << layer_id 
            << ": GELU..." << endl;
            
        nl.gelu(
            NL_NTHREADS,
            gelu_input,
            gelu_output,
            gelu_input_size,
            NL_ELL,
            NL_SCALE
        );

        cout << "-> Layer - " << layer_id 
            << ": GELU postprocessing..." << endl;

        lin.plain_col_packing_preprocess(
            gelu_output,
            gelu_output_col_pack,
            INPUT_DIM,
            INTER_DIM
        );

        vector<Ciphertext> h6 = ss_to_he_server(
            lin.he_8192, 
            gelu_output_col_pack,
            gelu_input_size);

        delete[] gelu_input_col_pack;
        delete[] gelu_input;
        delete[] gelu_output;
        delete[] gelu_output_col_pack;

        // ------------------ Linear #4 ------------------ //

        vector<Ciphertext> h7 = lin.linear_2(
            lin.he_8192,
            INPUT_DIM,
            3072,
            COMMON_DIM,
            h6, 
            w_i_2[layer_id],
            b_i_2[layer_id],
            data_lin4
        );

        cout << "-> Layer - " << layer_id << ": Linear Inter #2 done " << endl;

        // -------------------- Layer Norm -------------------- //
        
        int ln_2_input_size = INPUT_DIM*COMMON_DIM;
        uint64_t* ln_2_input_col_pack =
            new uint64_t[ln_2_input_size];
        uint64_t* ln_2_input =
            new uint64_t[ln_2_input_size];
        uint64_t* ln_2_output =
            new uint64_t[ln_2_input_size];
        uint64_t* ln_2_output_col_pack =
            new uint64_t[ln_2_input_size];
        
         // Secret sharing and send share to client
        he_to_ss_server(lin.he_8192, h7, ln_2_input_col_pack);
        cout << "-> Layer - " << layer_id 
            << ": Secret sharing Linear Inter #2 results done " << endl;

        // Post Processing
        lin.plain_col_packing_postprocess(
            ln_2_input_col_pack,
            ln_2_input,
            data_lin4
        );

        // H8 = Linear#4 + H4
        for(int i = 0; i < ln_2_input_size; i++){
            ln_2_input[i] += h4_cache[i];
        }

        nl.layer_norm(
            NL_NTHREADS,
            ln_2_input,
            ln_2_output,
            INPUT_DIM,
            COMMON_DIM,
            NL_ELL,
            NL_SCALE
        );

        // update H1
        memcpy(h1_cache, ln_2_output, ln_2_input_size*sizeof(uint64_t));

        lin.plain_col_packing_preprocess(
            ln_2_output,
            ln_2_output_col_pack,
            INPUT_DIM,
            COMMON_DIM
        );

        h1 = ss_to_he_server(
            lin.he_8192, 
            ln_2_output_col_pack,
            INPUT_DIM*COMMON_DIM);
    }

    // -------------------- POOL -------------------- //

    // -------------------- TANH -------------------- //

}

void Bert::run_client() {
    cout << "> Loading input" << endl;
    // Loading inputs 
    // H_1: 128×768
    vector<vector<uint64_t>> h1 = read_data("./txt/random_X.txt");

    uint64_t h1_cache[INPUT_DIM*COMMON_DIM] = {0};
    uint64_t h4_cache[INPUT_DIM*COMMON_DIM] = {0};

    // Column Packing
    vector<uint64_t> h1_vec(COMMON_DIM * INPUT_DIM);
    for (int j = 0; j < COMMON_DIM; j++){
        for (int i = 0; i < INPUT_DIM; i++){
            h1_vec[j*INPUT_DIM + i] = h1[i][j];
            h1_cache[i*COMMON_DIM + j] = h1[i][j];
        }
    }
            

    vector<Ciphertext> h1_cts = 
        lin.bert_efficient_preprocess_vec(lin.he_8192, h1_vec, data_lin1);
    
    send_encrypted_vector(io, h1_cts);

    cout << "> --- Entering Attention Layers ---" << endl;
    for(int layer_id; layer_id < ATTENTION_LAYERS; ++layer_id){

        // -------------- Waiting Linear#1 -------------- //
        int qk_size = PACKING_NUM*INPUT_DIM*INPUT_DIM;
        int v_size = PACKING_NUM*INPUT_DIM*OUTPUT_DIM;
        int softmax_size = PACKING_NUM*INPUT_DIM*INPUT_DIM;
        int att_size = PACKING_NUM*INPUT_DIM*OUTPUT_DIM;
        
        int qk_v_size = qk_size + v_size;
        int softmax_cts_len = qk_v_size / lin.he_8192->poly_modulus_degree;

        uint64_t* qk_v_col_pack = new uint64_t[qk_v_size];
        uint64_t* v_matrix = new uint64_t[v_size];
        uint64_t* softmax_input = new uint64_t[qk_size];
        uint64_t* softmax_output = new uint64_t[softmax_size];
        uint64_t* softmax_v = new uint64_t[att_size];
        
        // Secret sharing and get share from server
        cout << "-> Layer - " << layer_id << ": Secret Sharing" << endl;
        he_to_ss_client(lin.he_8192, qk_v_col_pack, softmax_cts_len, data_lin1);

        cout << "-> Layer - " << layer_id 
            << ": Softmax preprocessing..." << endl;

        // mod p
        nl.gt_p_sub(
            NL_NTHREADS,
            qk_v_col_pack,
            lin.he_8192->plain_mod,
            qk_v_col_pack,
            qk_v_size,
            NL_ELL,
            NL_SCALE
        );

        lin.plain_cross_packing_postprocess(
            qk_v_col_pack, 
            softmax_input,
            data_lin1);
        
        lin.plain_cross_packing_postprocess_v(
            &qk_v_col_pack[qk_size], 
            v_matrix,
            data_lin1);

        // -------------------- Softmax -------------------- //

        cout << "-> Layer - " << layer_id 
            << ": Softmax and multiply V..." << endl;
        // Softmax
        nl.softmax(
            NL_NTHREADS,
            softmax_input,
            softmax_output,
            12*INPUT_DIM,
            INPUT_DIM,
            NL_ELL,
            NL_SCALE);

        nl.n_matrix_mul(
            NL_NTHREADS,
            softmax_output,
            v_matrix,
            softmax_v,
            PACKING_NUM,
            INPUT_DIM,
            INPUT_DIM,
            OUTPUT_DIM,
            NL_ELL,
            NL_SCALE
        );
        cout << "-> Layer - " << layer_id 
            << ": Softmax postprocessing..." << endl;

        vector<vector<uint64_t>> h2_vec = lin.concat(softmax_v, 12, 128, 64);

        uint64_t* h2_col_packing = new uint64_t[att_size];
        // Packing before send back to server
        lin.plain_col_packing_preprocess_vec(
            h2_vec,
            h2_col_packing,
            INPUT_DIM,
            COMMON_DIM
        );

        ss_to_he_client(lin.he_8192, h2_col_packing, att_size);

        // Clean up
        delete [] qk_v_col_pack;
        delete [] softmax_input;
        delete [] v_matrix;
        delete [] softmax_output;
        delete [] softmax_v;
        delete [] h2_col_packing;

        // -------------- Waiting Linear#2 -------------- //
        int ln_size = INPUT_DIM*COMMON_DIM;
        int ln_cts_size = ln_size / lin.he_8192->poly_modulus_degree;
        uint64_t* ln_input_col_pack = new uint64_t[ln_size];
        uint64_t* ln_input = new uint64_t[ln_size];
        uint64_t* ln_output = new uint64_t[ln_size];
        uint64_t* ln_output_col_pack = new uint64_t[ln_size];
        
        // Secret sharing and get share from server
        cout << "-> Layer - " << layer_id << ": Secret Sharing" << endl;
        he_to_ss_client(lin.he_8192, ln_input_col_pack, ln_cts_size, data_lin2);

        cout << "-> Layer - " << layer_id 
            << ": Layer Norm preprocessing..." << endl;
        // Post Processing
        lin.plain_col_packing_postprocess(
            ln_input_col_pack,
            ln_input,
            data_lin2
        );

        // -------------------- Layer Norm -------------------- //

        // H3 = Linear#2 + H1
        for(int i = 0; i < ln_size; i++){
            ln_input[i] += h1_cache[i];
        }

        cout << "-> Layer - " << layer_id 
            << ": Layer Norm..." << endl;
        nl.layer_norm(
            NL_NTHREADS,
            ln_input,
            ln_output,
            INPUT_DIM,
            COMMON_DIM,
            NL_ELL,
            NL_SCALE
        );

        // update H4
        memcpy(h4_cache, ln_output, ln_size*sizeof(uint64_t));

        cout << "-> Layer - " << layer_id 
            << ": Layer Norm postprocessing..." << endl;
        lin.plain_col_packing_preprocess(
            ln_output,
            ln_output_col_pack,
            INPUT_DIM,
            COMMON_DIM
        );

        ss_to_he_client(lin.he_8192, ln_output_col_pack, ln_size);

        delete[] ln_input_col_pack;
        delete[] ln_input;
        delete[] ln_output;
        delete[] ln_output_col_pack;

        // -------------- Waiting Linear#3 -------------- //

        int gelu_input_size = 128*3072;
        int gelu_cts_size = gelu_input_size / lin.he_8192->poly_modulus_degree;
        uint64_t* gelu_input_col_pack =
            new uint64_t[gelu_input_size];
        uint64_t* gelu_input =
            new uint64_t[gelu_input_size];
        uint64_t* gelu_output =
            new uint64_t[gelu_input_size];
        uint64_t* gelu_output_col_pack =
            new uint64_t[gelu_input_size];

        // Secret sharing and get share from server
        cout << "-> Layer - " << layer_id << ": Secret Sharing" << endl;
        he_to_ss_client(lin.he_8192, gelu_input_col_pack, gelu_cts_size, data_lin3);
        cout << "-> Layer - " << layer_id 
            << ": GELU preprocessing..." << endl;

        // Post Processing
        lin.plain_col_packing_postprocess(
            gelu_input_col_pack,
            gelu_input,
            data_lin3
        );

        // ---------------------- GELU ---------------------- //

        cout << "-> Layer - " << layer_id 
            << ": GELU..." << endl;
        nl.gelu(
            NL_NTHREADS,
            gelu_input,
            gelu_output,
            gelu_input_size,
            NL_ELL,
            NL_SCALE
        );

        cout << "-> Layer - " << layer_id 
            << ": GELU postprocessing..." << endl;

        lin.plain_col_packing_preprocess(
            gelu_output,
            gelu_output_col_pack,
            INPUT_DIM,
            INTER_DIM);

        ss_to_he_client(
            lin.he_8192, 
            gelu_output_col_pack, 
            gelu_input_size);

        delete[] gelu_input_col_pack;
        delete[] gelu_input;
        delete[] gelu_output;
        delete[] gelu_output_col_pack;

        // -------------- Waiting Linear#4 -------------- //
        int ln_2_input_size = INPUT_DIM*COMMON_DIM;
        int ln_2_cts_size = ln_2_input_size/lin.he_8192->poly_modulus_degree;

        uint64_t* ln_2_input_col_pack =
            new uint64_t[ln_2_input_size];
        uint64_t* ln_2_input =
            new uint64_t[ln_2_input_size];
        uint64_t* ln_2_output =
            new uint64_t[ln_2_input_size];
        uint64_t* ln_2_output_col_pack =
            new uint64_t[ln_2_input_size];
        
        // Secret sharing and get share from server
        cout << "-> Layer - " << layer_id << ": Secret Sharing" << endl;
        he_to_ss_client(lin.he_8192, ln_2_input_col_pack, ln_2_cts_size, data_lin4);
        cout << "-> Layer - " << layer_id 
            << ": GELU preprocessing..." << endl;

        // Post Processing
        lin.plain_col_packing_postprocess(
            ln_2_input_col_pack,
            ln_2_input,
            data_lin4
        );

        // -------------------- Layer Norm -------------------- //

        // H8 = Linear#4 + H4
        for(int i = 0; i < ln_2_input_size; i++){
            ln_2_input[i] += h4_cache[i];
        }

        cout << "-> Layer - " << layer_id 
            << ": GELU..." << endl;
        nl.layer_norm(
            NL_NTHREADS,
            ln_2_input,
            ln_2_output,
            INPUT_DIM,
            COMMON_DIM,
            NL_ELL,
            NL_SCALE
        );

        // update H1
        memcpy(h1_cache, ln_2_output, ln_2_input_size*sizeof(uint64_t));

        cout << "-> Layer - " << layer_id 
            << ": GELU postprocessing..." << endl;
        lin.plain_col_packing_preprocess(
            ln_2_output,
            ln_2_output_col_pack,
            INPUT_DIM,
            COMMON_DIM
        );

        ss_to_he_client(
            lin.he_8192, 
            ln_2_output_col_pack, 
            ln_2_input_size);

        delete[] ln_2_input_col_pack;
        delete[] ln_2_input;
        delete[] ln_2_output;
        delete[] ln_2_output_col_pack;
    }


}