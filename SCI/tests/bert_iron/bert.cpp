#include "bert.h"

inline double interval(chrono::_V2::system_clock::time_point start){
    auto end = high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    return interval.count();
}

void save_to_file(uint64_t* matrix, size_t rows, size_t cols, const char* filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file << (int64_t)matrix[i * cols + j];
            if (j != cols - 1) {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}

void save_to_file_vec(vector<vector<uint64_t>> matrix, size_t rows, size_t cols, const char* filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file << matrix[i][j];
            if (j != cols - 1) {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}


Bert::Bert(int party, int port, string address, string model_path){
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new NetIO(party == 1 ? nullptr : address.c_str(), port);
    this->conv_err = 0;

    cout << "> Setup Linear" << endl;
    this->lin = Linear(party, io);
    cout << "> Setup NonLinear" << endl;
    this->nl = NonLinear(party, address, port+1);

    if(party == ALICE){
        cout << "> Loading and preprocessing weights on server" << endl;
        bm = load_model(model_path, NUM_CLASS);
        // lin.weights_preprocess(bm);
    }
    cout << "> Bert intialized done!" << endl << endl;

}

Bert::~Bert() {
    
}

void Bert::he_to_ss_server_plain(HE* he, uint64_t* input, uint64_t* output, int length){
    PRG128 prg;
    prg.random_mod_p<uint64_t>(output, length, he->plain_mod);

    uint64_t* tmp = new uint64_t[length];

    for(int i = 0; i < length; i++){
        tmp[i] = input[i] - output[i] + he->plain_mod_2;
        tmp[i] = neg_mod((int64_t)tmp[i], (int64_t) he->plain_mod);
    }
    io->send_data(tmp, length*sizeof(uint64_t));
    delete[] tmp;
}

void Bert::he_to_ss_client_plain(HE* he, uint64_t* output, int length){
    io->recv_data(output, length*sizeof(uint64_t));
}

void Bert::ss_to_he_server_plain(HE* he, uint64_t* input, int length, uint64_t* output){
    uint64_t* tmp = new uint64_t[length];
    uint64_t mask_v = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    io->recv_data(tmp, length*sizeof(uint64_t));
    for(int i = 0; i < length; i++){
        uint64_t v = input[i] + tmp[i];
        output[i] = neg_mod((int64_t)(v), (int64_t) he->plain_mod);
        if(output[i] > he->plain_mod_2){
            output[i] -= he->plain_mod;
        }
        v = v & mask_v;
        if(output[i] != v){
            conv_err += 1;
        }
    }
    delete[] tmp;
}

void Bert::ss_to_he_client_plain(HE* he, uint64_t* input, int length){
    io->send_data(input, length*sizeof(uint64_t));
}

void Bert::ln_share_server(
    int layer_id,
    vector<uint64_t> &wln_input,
    vector<uint64_t> &bln_input,
    uint64_t* wln,
    uint64_t* bln
){
    int length = 2*COMMON_DIM;
    uint64_t* random_share = new uint64_t[length];

    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    PRG128 prg;
    prg.random_data(random_share, length * sizeof(uint64_t));

    for(int i = 0; i < length; i++){
        random_share[i] &= mask_x;
    }

    io->send_data(random_share, length*sizeof(uint64_t));

    for(int i = 0; i < COMMON_DIM; i++){
        random_share[i] = (wln_input[i] -  random_share[i]) & mask_x;
        random_share[i + COMMON_DIM] = 
            (bln_input[i] -  random_share[i + COMMON_DIM]) & mask_x;
    }

    for(int i = 0; i < INPUT_DIM; i++){
        memcpy(&wln[i*COMMON_DIM], random_share, COMMON_DIM*sizeof(uint64_t));
        memcpy(&bln[i*COMMON_DIM], &random_share[COMMON_DIM], COMMON_DIM*sizeof(uint64_t));
    }

    delete[] random_share;
}

void Bert::ln_share_client(
    uint64_t* wln,
    uint64_t* bln
){
    int length = 2*COMMON_DIM;

    uint64_t* share = new uint64_t[length];
    io->recv_data(share, length * sizeof(uint64_t));
    for(int i = 0; i < INPUT_DIM; i++){
        memcpy(&wln[i*COMMON_DIM], share, COMMON_DIM*sizeof(uint64_t));
        memcpy(&bln[i*COMMON_DIM], &share[COMMON_DIM], COMMON_DIM*sizeof(uint64_t));
    }
    delete[] share;
}

void Bert::pc_bw_share_server(
    uint64_t* wp,
    uint64_t* bp,
    uint64_t* wc,
    uint64_t* bc
    ){
    int wp_len = COMMON_DIM*COMMON_DIM;
    int bp_len = COMMON_DIM;
    int wc_len = COMMON_DIM*NUM_CLASS;
    int bc_len = NUM_CLASS;

    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    int length =  wp_len + bp_len + wc_len + bc_len;
    uint64_t* random_share = new uint64_t[length];

    PRG128 prg;
    prg.random_data(random_share, length * sizeof(uint64_t));

    for(int i = 0; i < length; i++){
        random_share[i] &= mask_x;
    }

    io->send_data(random_share, length*sizeof(uint64_t));

    // cout << "share 1 " << endl;
    // Write wp share
    int offset = 0;
    for(int i = 0; i < COMMON_DIM; i++){
        for(int j = 0; j < COMMON_DIM; j++){
            wp[i*COMMON_DIM + j] = (bm.w_p[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }
    //  cout << "share 2 " << endl;
    // Write bp share
    for(int i = 0; i < COMMON_DIM; i++){
        bp[i] = (bm.b_p[i] - random_share[offset]) & mask_x;
        offset++;
    }
    //  cout << "share 3 " << endl;
    // Write w_c share
    for(int i = 0; i < COMMON_DIM; i++){
        for(int j = 0; j < NUM_CLASS; j++){
            wc[i*NUM_CLASS + j] = (bm.w_c[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }
    //  cout << "share 4 " << endl;
    // Write b_c share
    for(int i = 0; i < NUM_CLASS; i++){
        bc[i] = (bm.b_c[i]- random_share[offset]) & mask_x;
        offset++;
    } 
    delete[] random_share;
}

void Bert::pc_bw_share_client(
    uint64_t* wp,
    uint64_t* bp,
    uint64_t* wc,
    uint64_t* bc
    ){
    int wp_len = COMMON_DIM*COMMON_DIM;
    int bp_len = COMMON_DIM;
    int wc_len = COMMON_DIM*NUM_CLASS;
    int bc_len = NUM_CLASS; 
    int length =  wp_len + bp_len + wc_len + bc_len;  

    uint64_t* share = new uint64_t[length];
    io->recv_data(share, length * sizeof(uint64_t));
    memcpy(wp, share, wp_len*sizeof(uint64_t));
    memcpy(bp, &share[wp_len], bp_len*sizeof(uint64_t));
    memcpy(wc, &share[wp_len + bp_len], wc_len*sizeof(uint64_t));
    memcpy(bc, &share[wp_len + bp_len + wc_len], bc_len*sizeof(uint64_t));
    delete[] share;
}

vector<double> Bert::run(string input_fname, string mask_fname){
    // Server: Alice
    // Client: Bob

    vector<uint64_t> softmax_mask;

    uint64_t h1_cache_12[INPUT_DIM*COMMON_DIM] = {0};
    uint64_t h4_cache_12[INPUT_DIM*COMMON_DIM] = {0};
    uint64_t h98[COMMON_DIM] = {0};
    uint64_t h1[INPUT_DIM*COMMON_DIM];
    uint64_t h2[INPUT_DIM*COMMON_DIM];
    uint64_t h4[INPUT_DIM*COMMON_DIM];
    uint64_t h6[INPUT_DIM*INTER_DIM];

    if(party == ALICE){
        // -------------------- Preparing -------------------- //
        // Receive cipher text input
        io->recv_data(h1, INPUT_DIM*COMMON_DIM*sizeof(uint64_t));
    } else{
        cout << "> Loading inputs" << endl;
        vector<vector<uint64_t>> input_plain = read_data(input_fname);
        softmax_mask = read_bias(mask_fname, 128);

        // Column Packing
        uint64_t input_row[COMMON_DIM * INPUT_DIM];
        for (int j = 0; j < COMMON_DIM; j++){
            for (int i = 0; i < INPUT_DIM; i++){
                // rescale input from 12 to 5
                input_row[i*COMMON_DIM + j] = ((int64_t)input_plain[i][j]) >> 7;
                h1_cache_12[i*COMMON_DIM + j] = input_plain[i][j];
            }
        }

        // Send plain text input
        io->send_data(input_row, INPUT_DIM*COMMON_DIM*sizeof(uint64_t));
    }

    cout << "> --- Entering Attention Layers ---" << endl;
    for(int layer_id; layer_id < ATTENTION_LAYERS; ++layer_id){
        {
            // -------------------- Linear #1 -------------------- //
            int qk_size = PACKING_NUM*INPUT_DIM*INPUT_DIM;
            int v_size = PACKING_NUM*INPUT_DIM*OUTPUT_DIM;
            int softmax_size = PACKING_NUM*INPUT_DIM*INPUT_DIM;
            int att_size = PACKING_NUM*INPUT_DIM*OUTPUT_DIM;  
            int qk_v_size = qk_size + v_size;

            uint64_t* q_matrix_row = new uint64_t[v_size];
            uint64_t* k_trans_matrix_row = new uint64_t[v_size];
            uint64_t* v_matrix_row = new uint64_t[v_size];

            uint64_t* softmax_input_row = new uint64_t[qk_size];
            uint64_t* softmax_output_row = new uint64_t[softmax_size];
            uint64_t* softmax_v_row = new uint64_t[att_size];
            uint64_t* h2_concate = new uint64_t[att_size];
                    
            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #1 HE" << endl;
                linear_1_plain(
                    h1,
                    bm.w_q[layer_id],
                    bm.w_k[layer_id],
                    bm.w_v[layer_id],
                    bm.b_q[layer_id],
                    bm.b_k[layer_id],
                    bm.b_v[layer_id],
                    q_matrix_row,
                    k_trans_matrix_row,
                    v_matrix_row
                );
                cout << "-> Layer - " << layer_id << ": Linear #1 done HE" << endl;
            } else{
                for(int i = 0; i < v_size; i++){
                    q_matrix_row[i] = 0;
                    k_trans_matrix_row[i] = 0;
                    v_matrix_row[i] = 0;
                }
            }

            nl.right_shift(
                NL_NTHREADS,
                q_matrix_row,
                NL_SCALE,
                q_matrix_row,
                v_size,
                NL_ELL,
                2*NL_SCALE
            );

            nl.right_shift(
                NL_NTHREADS,
                k_trans_matrix_row,
                NL_SCALE,
                k_trans_matrix_row,
                v_size,
                NL_ELL,
                2*NL_SCALE
            );

            nl.right_shift(
                NL_NTHREADS,
                v_matrix_row,
                NL_SCALE,
                v_matrix_row,
                v_size,
                NL_ELL,
                2*NL_SCALE
            );

            nl.n_matrix_mul_iron(
                NL_NTHREADS,
                q_matrix_row,
                k_trans_matrix_row,
                softmax_input_row,
                PACKING_NUM,
                INPUT_DIM,
                OUTPUT_DIM,
                INPUT_DIM,
                NL_ELL,
                NL_SCALE,
                NL_SCALE,
                NL_SCALE
            );

            if (party == BOB){
                // Add mask
                for(int i = 0; i < PACKING_NUM; i++){
                    int offset_nm = i*INPUT_DIM*INPUT_DIM;
                    for(int j = 0; j < INPUT_DIM; j++){
                        int offset_row = j*INPUT_DIM;
                        for (int k = 0; k < INPUT_DIM; k++){
                            softmax_input_row[offset_nm + offset_row + k] += 
                                softmax_mask[k] * 4096;
                        }
                    }
                }
            }

            // Softmax
            nl.softmax_iron(
                NL_NTHREADS,
                softmax_input_row,
                softmax_output_row,
                12*INPUT_DIM,
                INPUT_DIM,
                NL_ELL,
                NL_SCALE);

            // Rescale to 6
            nl.n_matrix_mul_iron(
                NL_NTHREADS,
                softmax_output_row,
                v_matrix_row,
                softmax_v_row,
                PACKING_NUM,
                INPUT_DIM,
                INPUT_DIM,
                OUTPUT_DIM,
                NL_ELL,
                NL_SCALE,
                NL_SCALE,
                NL_SCALE
            );

             lin.concat(softmax_v_row, h2_concate, 12, 128, 64);

            if(party == ALICE){
                io->recv_data(h2, att_size*sizeof(uint64_t));
                for(int i = 0; i < att_size; i++){
                    h2[i] += h2_concate[i];
                }
            } else{
                io->send_data(h2_concate, att_size*sizeof(uint64_t));
            }

            delete [] q_matrix_row;
            delete [] k_trans_matrix_row;
            delete [] v_matrix_row;
            delete [] softmax_input_row;
            delete [] softmax_output_row;
            delete [] softmax_v_row;
            delete [] h2_concate;
        }

        // -------------------- Linear #2 -------------------- //
        {
            int ln_size = INPUT_DIM*COMMON_DIM;
            uint64_t* ln_input_row = new uint64_t[ln_size];
            uint64_t* ln_output_row = new uint64_t[ln_size];
            uint64_t* ln_weight = new uint64_t[ln_size];
            uint64_t* ln_bias = new uint64_t[ln_size];

            
            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #2 HE" << endl;
                linear_2_plain(
                    h2,
                    bm.w_o[layer_id],
                    bm.b_o[layer_id],
                    128,
                    768,
                    768,
                    ln_input_row
                );
                cout << "-> Layer - " << layer_id << ": Linear #2 HE done " << endl;
                
                ln_share_server(
                    layer_id,
                    bm.w_ln_1[layer_id],
                    bm.b_ln_1[layer_id],
                    ln_weight,
                    ln_bias
                );
            } else{
                for(int i = 0; i < ln_size; i++){
                    ln_input_row[i] = 0;
                }
                ln_share_client(
                    ln_weight,
                    ln_bias
                );
            }

             nl.right_shift(
                NL_NTHREADS,
                ln_input_row,
                NL_SCALE,
                ln_input_row,
                ln_size,
                NL_ELL,
                2*NL_SCALE
            );

            for(int i = 0; i < ln_size; i++){
                ln_input_row[i] += h1_cache_12[i];
            }

            // Layer Norm
            nl.layer_norm(
                NL_NTHREADS,
                ln_input_row,
                ln_output_row,
                ln_weight,
                ln_bias,
                INPUT_DIM,
                COMMON_DIM,
                NL_ELL,
                NL_SCALE
            );
            
            memcpy(h4_cache_12, ln_output_row, ln_size*sizeof(uint64_t));

            if(party == ALICE){
                io->recv_data(h4, ln_size*sizeof(uint64_t));
                for(int i = 0; i < ln_size; i++){
                    h4[i] += ln_output_row[i];
                }
            } else{
                io->send_data(ln_output_row, ln_size*sizeof(uint64_t));
            }

            delete[] ln_input_row;
            delete[] ln_output_row;
            delete[] ln_weight;
            delete[] ln_bias;
        }
    

        // -------------------- Linear #3 -------------------- //
        {
            int gelu_input_size = 128*3072;
            uint64_t* gelu_input_col =
                new uint64_t[gelu_input_size];
            uint64_t* gelu_output_col =
                new uint64_t[gelu_input_size];

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #3 HE" << endl;
                linear_2_plain(
                    h4,
                    bm.w_i_1[layer_id],
                    bm.b_i_1[layer_id],
                    128,
                    768,
                    3072,
                    gelu_input_col
                );
                cout << "-> Layer - " << layer_id << ": Linear #3 HE done " << endl;
            } else{
                for(int i = 0; i < gelu_input_size; i++){
                    gelu_input_col[i] = 0;
                }
            }

            nl.right_shift(
                NL_NTHREADS,
                gelu_input_col,
                NL_SCALE,
                gelu_input_col,
                gelu_input_size,
                NL_ELL,
                2*NL_SCALE
            );

            nl.gelu_iron(
                NL_NTHREADS,
                gelu_input_col,
                gelu_output_col,
                gelu_input_size,
                NL_ELL,
                NL_SCALE
            );

            if(party == ALICE){
                io->recv_data(h6, gelu_input_size*sizeof(uint64_t));
                for(int i = 0; i < gelu_input_size; i++){
                    h6[i] += gelu_output_col[i];
                }

            } else{
                io->send_data(gelu_output_col, gelu_input_size*sizeof(uint64_t));
            }

            delete[] gelu_input_col;
            delete[] gelu_output_col;
        }

        {
            int ln_2_input_size = INPUT_DIM*COMMON_DIM;

            uint64_t* ln_2_input_row =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_output_row =
                new uint64_t[ln_2_input_size];

            uint64_t* ln_weight_2 = new uint64_t[ln_2_input_size];
            uint64_t* ln_bias_2 = new uint64_t[ln_2_input_size];

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #4 HE " << endl;
                linear_2_plain(
                    h6,
                    bm.w_i_2[layer_id],
                    bm.b_i_2[layer_id],
                    128,
                    3072,
                    768,
                    ln_2_input_row
                );
                cout << "-> Layer - " << layer_id << ": Linear #4 HE done" << endl;

                ln_share_server(
                    layer_id,
                    bm.w_ln_2[layer_id],
                    bm.b_ln_2[layer_id],
                    ln_weight_2,
                    ln_bias_2
                );
            } else{
                for(int i = 0; i < ln_2_input_size; i++){
                    ln_2_input_row[i] = 0;
                }

                ln_share_client(
                    ln_weight_2,
                    ln_bias_2
                );
            }

            nl.right_shift(
                NL_NTHREADS,
                ln_2_input_row,
                NL_SCALE,
                ln_2_input_row,
                ln_2_input_size,
                NL_ELL,
                2*NL_SCALE
            );

            for(int i = 0; i < ln_2_input_size; i++){
                ln_2_input_row[i] += h4_cache_12[i];
            }

            nl.layer_norm(
                NL_NTHREADS,
                ln_2_input_row,
                ln_2_output_row,
                ln_weight_2,
                ln_bias_2,
                INPUT_DIM,
                COMMON_DIM,
                NL_ELL,
                NL_SCALE
            );


            // update H1
            memcpy(h1_cache_12, ln_2_output_row, ln_2_input_size*sizeof(uint64_t));


            if(layer_id == 11){
                // Using Scale of 12 as 
                memcpy(h98, h1_cache_12, COMMON_DIM*sizeof(uint64_t));
            } else{
                if(party == ALICE){
                    io->recv_data(h1, ln_2_input_size*sizeof(uint64_t));
                    for(int i=0; i < ln_2_input_size; i++){
                        h1[i] += ln_2_output_row[i];
                    }
                } else{
                    io->send_data(ln_2_output_row, ln_2_input_size*sizeof(uint64_t));
                }
            }

            delete[] ln_2_input_row;
            delete[] ln_2_output_row;
            delete[] ln_weight_2;
            delete[] ln_bias_2;
        }
    }

    // Secret share Pool and Classification model
    uint64_t* wp = new uint64_t[COMMON_DIM*COMMON_DIM];
    uint64_t* bp = new uint64_t[COMMON_DIM];
    uint64_t* wc = new uint64_t[COMMON_DIM*NUM_CLASS];
    uint64_t* bc = new uint64_t[NUM_CLASS];

    uint64_t* h99 = new uint64_t[COMMON_DIM];
    uint64_t* h100 = new uint64_t[COMMON_DIM];
    uint64_t* h101 = new uint64_t[NUM_CLASS];

    cout << "-> Sharing Pooling and Classification params..." << endl;

    if(party == ALICE){
        pc_bw_share_server(
            wp,
            bp,
            wc,
            bc
        );
    } else{
        pc_bw_share_client(
            wp,
            bp,
            wc,
            bc
        );
    }

    #ifdef BERT_TIMING
    auto t_pc = high_resolution_clock::now();
    #endif 

    // -------------------- POOL -------------------- //
    cout << "-> Layer - Pooling" << endl;
    nl.n_matrix_mul_iron(
        NL_NTHREADS,
        h98,
        wp,
        h99,
        1,
        1,
        COMMON_DIM,
        COMMON_DIM,
        NL_ELL,
        NL_SCALE,
        NL_SCALE,
        2*NL_SCALE
    );

    #ifdef BERT_TIMING
    cout << "> [TIMING]: Pooling mul takes:" << interval(t_pc) << " sec" << endl; 
    #endif 

    for(int i = 0; i < COMMON_DIM; i++){
        h99[i] += bp[i];
    }

    nl.right_shift(
        NL_NTHREADS,
        h99,
        NL_SCALE,
        h99,
        COMMON_DIM,
        NL_ELL,
        2*NL_SCALE
    );

    #ifdef BERT_TIMING
    cout << "> [TIMING]: Pooling takes:" << interval(t_pc) << " sec" << endl; 
    #endif 

    // -------------------- TANH -------------------- //
    nl.tanh(
        NL_NTHREADS,
        h99,
        h100,
        COMMON_DIM,
        NL_ELL,
        NL_SCALE
    );
    
    cout << "-> Layer - Classification" << endl;
    nl.n_matrix_mul_iron(
        NL_NTHREADS,
        h100,
        wc,
        h101,
        1,
        1,
        COMMON_DIM,
        NUM_CLASS,
        NL_ELL,
        NL_SCALE,
        NL_SCALE,
        2*NL_SCALE
    );

    for(int i = 0; i < NUM_CLASS; i++){
        h101[i] += bc[i];
    }

    nl.right_shift(
        1,
        h101,
        NL_SCALE,
        h101,
        NUM_CLASS,
        NL_ELL,
        2*NL_SCALE
    );

    if(party == ALICE){
        io->send_data(h101, NUM_CLASS*sizeof(uint64_t));
        return {};
    } else{
        uint64_t* res = new uint64_t[NUM_CLASS];
        vector<double> dbl_result;
        io->recv_data(res, NUM_CLASS*sizeof(uint64_t));

        for(int i = 0; i < NUM_CLASS; i++){
            dbl_result.push_back((signed_val(res[i] + h101[i], NL_ELL)) / double(1LL << NL_SCALE));
        }
        return dbl_result;
    }
}