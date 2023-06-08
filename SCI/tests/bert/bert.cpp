#include "bert.h"

#ifdef BERT_TIMING
double t_total_linear1 = 0;
double t_total_linear2 = 0;
double t_total_linear3 = 0;
double t_total_linear4 = 0;

double t_total_softmax = 0;
double t_total_mul_v = 0;
double t_total_gelu = 0;
double t_total_ln_1 = 0;
double t_total_ln_2 = 0;

double t_total_repacking = 0;
double t_total_gt_sub = 0;
double t_total_shift = 0;

double t_total_conversion = 0;

double t_total_ln_share = 0;
#endif 

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

void print_pt(HE* he, Plaintext &pt, int len) {
    vector<uint64_t> dest(len, 0ULL);
    he->encoder->decode(pt, dest);
    cout << "Decode first 5 rows: ";
    int non_zero_count;
    for(int i = 0; i < 16; i++){
        if(dest[i] > he->plain_mod_2) {
            cout << (int64_t)(dest[i] - he->plain_mod) << " ";
        } else{
            cout << dest[i] << " ";
        }
        // if(dest[i] != 0){
        //     non_zero_count += 1;
        // }
    }
    // cout << "Non zero count: " << non_zero_count;
    cout << endl;
}

void print_ct(HE* he, Ciphertext &ct, int len){
    Plaintext pt;
    he->decryptor->decrypt(ct, pt);
    cout << "Noise budget: ";
    cout << YELLOW << he->decryptor->invariant_noise_budget(ct) << " ";
    cout << RESET << endl;
    print_pt(he, pt, len);
}

Bert::Bert(int party, int port, string address, string model_path){
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

    cout << "> Setup Linear" << endl;
    this->lin = Linear(party, io);
    cout << "> Setup NonLinear" << endl;
    this->nl = NonLinear(party, address, port+1);

    if(party == ALICE){
        cout << "> Loading and preprocessing weights on server" << endl;
        #ifdef BERT_TIMING
        auto t_load_model = high_resolution_clock::now();
        #endif 

        struct BertModel bm = 
            load_model(model_path, NUM_CLASS);

        #ifdef BERT_TIMING
        cout << "> [TIMING]: Loading Model takes: " << interval(t_load_model) << "sec" << endl;
        auto t_model_preprocess = high_resolution_clock::now();
        #endif 

        lin.weights_preprocess(bm);

        #ifdef BERT_TIMING
        cout << "> [TIMING]: Model Preprocessing takes: " << interval(t_model_preprocess) << "sec" << endl;
        #endif 
    }
    cout << "> Bert intialized done!" << endl << endl;

}

Bert::~Bert() {
    
}

void Bert::he_to_ss_server(HE* he, vector<Ciphertext> in, uint64_t* output){
    #ifdef BERT_TIMING
    auto t_conversion = high_resolution_clock::now();
    #endif 

    PRG128 prg;
    int dim = in.size();
    int slot_count = he->poly_modulus_degree;
	// prg.random_mod_p<uint64_t>(output, dim*slot_count, he->plain_mod);
    for(int i = 0; i < dim*slot_count; i++){
        output[i] = 0;
    }

    Plaintext pt_p_2;
    vector<uint64_t> p_2(slot_count, he->plain_mod_2);
    he->encoder->encode(p_2, pt_p_2);

    vector<Ciphertext> cts;
    for(int i = 0; i < dim; i++){
        vector<uint64_t> tmp(slot_count);
        for(int j = 0; j < slot_count; ++j){
            tmp[j] = output[i*slot_count + j];
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        Ciphertext ct; 
        he->evaluator->sub_plain(in[i], pt, ct);
        he->evaluator->add_plain_inplace(ct, pt_p_2);
        // print_pt(he, pt, 8192);
        cts.push_back(ct);
    }
    send_encrypted_vector(io, cts);
    #ifdef BERT_TIMING
    t_total_conversion += interval(t_conversion);
    #endif 
}

vector<Ciphertext> Bert::ss_to_he_server(HE* he, uint64_t* input, int length){
    #ifdef BERT_TIMING
    auto t_conversion = high_resolution_clock::now();
    #endif 
    int slot_count = he->poly_modulus_degree;
    uint64_t plain_mod = he->plain_mod;
    vector<Plaintext> share_server;
    int dim = length / slot_count;
    for(int i = 0; i < dim; i++){
        vector<uint64_t> tmp(slot_count);
        for(int j = 0; j < slot_count; ++j){
            tmp[j] = neg_mod((int64_t)input[i*slot_count + j], (int64_t)plain_mod);
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
    #ifdef BERT_TIMING
    t_total_conversion += interval(t_conversion);
    #endif 
    return share_client;
}

void Bert::he_to_ss_client(HE* he, uint64_t* output, int length, const FCMetadata &data){
    #ifdef BERT_TIMING
    auto t_conversion = high_resolution_clock::now();
    #endif 
    vector<Ciphertext> cts(length);
    recv_encrypted_vector(he->context, io, cts);
    for(int i = 0; i < length; i++){
        vector<uint64_t> plain(data.slot_count, 0ULL);
        Plaintext tmp;
        he->decryptor->decrypt(cts[i], tmp);
        he->encoder->decode(tmp, plain);
        std::copy(plain.begin(), plain.end(), &output[i*data.slot_count]);
    }
    #ifdef BERT_TIMING
    t_total_conversion += interval(t_conversion);
    #endif 
}

void Bert::ss_to_he_client(HE* he, uint64_t* input, int length){
    #ifdef BERT_TIMING
    auto t_conversion = high_resolution_clock::now();
    #endif 
    int slot_count = he->poly_modulus_degree;
    uint64_t plain_mod = he->plain_mod;
    vector<Ciphertext> cts;
    int dim = length / slot_count;
    for(int i = 0; i < dim; i++){
        vector<uint64_t> tmp(slot_count);
        for(int j = 0; j < slot_count; ++j){
             tmp[j] = neg_mod((int64_t)input[i*slot_count + j], (int64_t)plain_mod);
        }
        Plaintext pt;
        he->encoder->encode(tmp, pt);
        Ciphertext ct; 
        he->encryptor->encrypt(pt, ct);
        cts.push_back(ct);
    }
    send_encrypted_vector(io, cts);
    #ifdef BERT_TIMING
    t_total_conversion += interval(t_conversion);
    #endif 
}

void Bert::ln_share_server(
    int layer_id,
    vector<uint64_t> &wln_input,
    vector<uint64_t> &bln_input,
    uint64_t* wln,
    uint64_t* bln
){
    #ifdef BERT_TIMING
    auto t_ln_share = high_resolution_clock::now();
    #endif 

    int length = 2*COMMON_DIM;
    uint64_t* random_share = new uint64_t[length];

    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    // PRG128 prg;
    // prg.random_data(random_share, length * sizeof(uint64_t));

    // for(int i = 0; i < length; i++){
    //     random_share[i] &= mask_x;
    // }

    for(int i = 0; i < length; i++){
        random_share[i] = 0;
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
    #ifdef BERT_TIMING
    t_total_ln_share += interval(t_ln_share);
    #endif 
}

void Bert::ln_share_client(
    uint64_t* wln,
    uint64_t* bln
){

    #ifdef BERT_TIMING
    auto t_ln_share = high_resolution_clock::now();
    #endif 

    int length = 2*COMMON_DIM;

    uint64_t* share = new uint64_t[length];
    io->recv_data(share, length * sizeof(uint64_t));
    for(int i = 0; i < INPUT_DIM; i++){
        memcpy(&wln[i*COMMON_DIM], share, COMMON_DIM*sizeof(uint64_t));
        memcpy(&bln[i*COMMON_DIM], &share[COMMON_DIM], COMMON_DIM*sizeof(uint64_t));
    }
    delete[] share;
    #ifdef BERT_TIMING
    t_total_ln_share += interval(t_ln_share);
    #endif 
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

    // PRG128 prg;
    // prg.random_data(random_share, length * sizeof(uint64_t));

    // for(int i = 0; i < length; i++){
    //     random_share[i] &= mask_x;
    // }

    for(int i = 0; i < length; i++){
        random_share[i] = 0;
    }

    io->send_data(random_share, length*sizeof(uint64_t));

    // cout << "share 1 " << endl;
    // Write wp share
    int offset = 0;
    for(int i = 0; i < COMMON_DIM; i++){
        for(int j = 0; j < COMMON_DIM; j++){
            wp[i*COMMON_DIM + j] = (lin.w_p[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }
    //  cout << "share 2 " << endl;
    // Write bp share
    for(int i = 0; i < COMMON_DIM; i++){
        bp[i] = (lin.b_p[i] - random_share[offset]) & mask_x;
        offset++;
    }
    //  cout << "share 3 " << endl;
    // Write w_c share
    for(int i = 0; i < COMMON_DIM; i++){
        for(int j = 0; j < NUM_CLASS; j++){
            wc[i*NUM_CLASS + j] = (lin.w_c[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }
    //  cout << "share 4 " << endl;
    // Write b_c share
    for(int i = 0; i < NUM_CLASS; i++){
        bc[i] = (lin.b_c[i]- random_share[offset]) & mask_x;
        offset++;
    } 
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
}

vector<double> Bert::run(string input_fname, string mask_fname){
    // Server: Alice
    // Client: Bob

    vector<uint64_t> softmax_mask;
    uint64_t h1_cache_12[INPUT_DIM*COMMON_DIM] = {0};
    uint64_t h4_cache_12[INPUT_DIM*COMMON_DIM] = {0};
    uint64_t h98[COMMON_DIM] = {0};
    vector<Ciphertext> h1(12);
    vector<Ciphertext> h2;
    vector<Ciphertext> h4;
    vector<Ciphertext> h6;

    if(party == ALICE){
        // -------------------- Preparing -------------------- //
        // Receive cipher text input
        recv_encrypted_vector(lin.he_8192->context, io, h1);
        cout << "> Receive input cts from client " << endl;
    } else{
        cout << "> Loading inputs" << endl;
        vector<vector<uint64_t>> input_plain = read_data(input_fname);
        softmax_mask = read_bias(mask_fname, 128);

        // Column Packing
        vector<uint64_t> input_col(COMMON_DIM * INPUT_DIM);
        for (int j = 0; j < COMMON_DIM; j++){
            for (int i = 0; i < INPUT_DIM; i++){
                input_col[j*INPUT_DIM + i] = neg_mod((int64_t)input_plain[i][j], (int64_t)lin.he_8192->plain_mod);
                h1_cache_12[i*COMMON_DIM + j] = input_plain[i][j] << 7;
            }
        }

        // Send cipher text input
        vector<Ciphertext> h1_cts = 
            lin.bert_efficient_preprocess_vec(lin.he_8192, input_col, lin.data_lin1);
        send_encrypted_vector(io, h1_cts);
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
            uint64_t* qk_v_cross = new uint64_t[qk_v_size];
            uint64_t* v_matrix_row = new uint64_t[v_size];
            uint64_t* softmax_input_row = new uint64_t[qk_size];
            uint64_t* softmax_output_row = new uint64_t[softmax_size];
            uint64_t* softmax_v_row = new uint64_t[att_size];
            uint64_t* h2_concate = new uint64_t[att_size];
            uint64_t* h2_col = new uint64_t[att_size];
                    
            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #1 HE" << endl;
                #ifdef BERT_TIMING
                auto t_linear1 = high_resolution_clock::now();
                #endif 
                vector<Ciphertext> q_k_v = lin.linear_1(
                    lin.he_8192,
                    h1,
                    lin.pp_1[layer_id],
                    lin.data_lin1
                );
                #ifdef BERT_TIMING
                t_total_linear1 += interval(t_linear1);
                #endif 
                cout << "-> Layer - " << layer_id << ": Linear #1 done HE" << endl;
                he_to_ss_server(lin.he_8192, q_k_v, qk_v_cross);
            } else{
                int softmax_cts_len = qk_v_size / lin.he_8192->poly_modulus_degree;
                he_to_ss_client(lin.he_8192, qk_v_cross, softmax_cts_len, lin.data_lin1);
            }

            #ifdef BERT_TIMING
            auto t_gt_sub = high_resolution_clock::now();
            #endif 

            // Scale: Q*V 22 V 11
            nl.gt_p_sub(
                NL_NTHREADS,
                qk_v_cross,
                lin.he_8192->plain_mod,
                qk_v_cross,
                qk_v_size,
                NL_ELL,
                22,
                22
            );

            #ifdef BERT_TIMING
            t_total_gt_sub += interval(t_gt_sub);
            auto t_repacking = high_resolution_clock::now();
            #endif 

            lin.plain_cross_packing_postprocess(
                qk_v_cross, 
                softmax_input_row,
                // we need row packing
                false,
                lin.data_lin1);
            
            lin.plain_cross_packing_postprocess_v(
                &qk_v_cross[qk_size], 
                v_matrix_row,
                false,
                lin.data_lin1);

            #ifdef BERT_TIMING
            t_total_repacking += interval(t_repacking);
            auto t_shift = high_resolution_clock::now();
            #endif 

            
            // Rescale QK to 12
            nl.right_shift(
                NL_NTHREADS,
                softmax_input_row,
                22 - NL_SCALE,
                softmax_input_row,
                qk_size,
                NL_ELL,
                NL_SCALE
            );

            // for(int i = 0; i < qk_size; i++){
            //     softmax_input_row[i] = ((int64_t)softmax_input_row[i]) >> 10;
            // }

            // nl.print_ss(softmax_input_row, 16, NL_ELL, NL_SCALE);
            // return {};

            #ifdef BERT_TIMING
            t_total_shift += interval(t_shift);
            auto t_softmax = high_resolution_clock::now();
            #endif 


            if (party == BOB){
                // Add mask
                for(int i = 0; i < PACKING_NUM; i++){
                    int offset_nm = i*INPUT_DIM*INPUT_DIM;
                    for(int j = 0; j < INPUT_DIM; j++){
                        int offset_row = j*INPUT_DIM;
                        for (int k = 0; k < INPUT_DIM; k++){
                            softmax_input_row[offset_nm + offset_row + k] += 
                                softmax_mask[k] * 100;
                        }
                    }
                }
            }

            // Softmax
            nl.softmax(
                NL_NTHREADS,
                softmax_input_row,
                softmax_output_row,
                12*INPUT_DIM,
                INPUT_DIM,
                NL_ELL,
                NL_SCALE);
            
            #ifdef BERT_TIMING
            t_total_softmax += interval(t_softmax);
            auto t_mul_v = high_resolution_clock::now();
            #endif 


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
                11,
                6
            );
            lin.concat(softmax_v_row, h2_concate, 12, 128, 64); 

            #ifdef BERT_TIMING
            t_total_mul_v += interval(t_mul_v);
            auto t_repacking_2 = high_resolution_clock::now();
            #endif 


            lin.plain_col_packing_preprocess(
                h2_concate,
                h2_col,
                lin.he_8192_tiny->plain_mod,
                INPUT_DIM,
                COMMON_DIM
            );

            #ifdef BERT_TIMING
            t_total_repacking += interval(t_repacking_2);
            #endif 


            if(party == ALICE){
                h2 = ss_to_he_server(
                    lin.he_8192_tiny, 
                    h2_col,
                    att_size);
            } else{
                ss_to_he_client(lin.he_8192_tiny, h2_col, att_size);
            }
            delete [] qk_v_cross;
            delete [] v_matrix_row;
            delete [] softmax_input_row;
            delete [] softmax_output_row;
            delete [] softmax_v_row;
            delete [] h2_concate;
            delete [] h2_col;
        }

        // -------------------- Linear #2 -------------------- //
        {
            int ln_size = INPUT_DIM*COMMON_DIM;
            int ln_cts_size = ln_size / lin.he_8192_tiny->poly_modulus_degree;
            uint64_t* ln_input_cross = new uint64_t[ln_size];
            uint64_t* ln_input_row = new uint64_t[ln_size];
            uint64_t* ln_output_row = new uint64_t[ln_size];
            uint64_t* ln_output_col = new uint64_t[ln_size];
            uint64_t* ln_weight = new uint64_t[ln_size];
            uint64_t* ln_bias = new uint64_t[ln_size];

            
            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #2 HE" << endl;
                #ifdef BERT_TIMING
                auto t_linear2 = high_resolution_clock::now();
                #endif 
                vector<Ciphertext> h3 = lin.linear_2(
                    lin.he_8192_tiny,
                    h2, 
                    lin.pp_2[layer_id],
                    lin.data_lin2
                );
                #ifdef BERT_TIMING
                t_total_linear2 += interval(t_linear2);
                #endif 
                cout << "-> Layer - " << layer_id << ": Linear #2 HE done " << endl;
                he_to_ss_server(lin.he_8192_tiny, h3, ln_input_cross);
                ln_share_server(
                    layer_id,
                    lin.w_ln_1[layer_id],
                    lin.b_ln_1[layer_id],
                    ln_weight,
                    ln_bias
                );
            } else{
                vector<Ciphertext> h3(ln_cts_size);
                he_to_ss_client(lin.he_8192_tiny, ln_input_cross, ln_cts_size, lin.data_lin2);
                ln_share_client(
                    ln_weight,
                    ln_bias
                );
            }

            #ifdef BERT_TIMING
            auto t_repacking = high_resolution_clock::now();
            #endif 

            lin.plain_col_packing_postprocess(
                ln_input_cross,
                ln_input_row,
                false,
                lin.data_lin2
            );

            #ifdef BERT_TIMING
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
            #endif 

            nl.gt_p_sub(
                NL_NTHREADS,
                ln_input_row,
                lin.he_8192_tiny->plain_mod,
                ln_input_row,
                ln_size,
                NL_ELL,
                NL_SCALE,
                NL_SCALE
            );

            #ifdef BERT_TIMING
            t_total_gt_sub += interval(t_gt_sub);
            auto t_ln_1 = high_resolution_clock::now();
            #endif 

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

            #ifdef BERT_TIMING
            t_total_ln_1 += interval(t_ln_1);
            auto t_shift = high_resolution_clock::now();
            #endif 

            memcpy(h4_cache_12, ln_output_row, ln_size*sizeof(uint64_t));

            nl.right_shift(
                NL_NTHREADS,
                ln_output_row,
                NL_SCALE - 5,
                ln_output_row,
                ln_size,
                64,
                NL_SCALE
            );

            #ifdef BERT_TIMING
            t_total_shift += interval(t_shift);
            auto t_repacking_2 = high_resolution_clock::now();
            #endif 

            // FixArray tmp = nl.to_public(ln_output_row, 128*768, 64, 5);
            // save_to_file(tmp.data, 128, 768, "./inter_result/linear3_input.txt");

            lin.plain_col_packing_preprocess(
                ln_output_row,
                ln_output_col,
                lin.he_8192_tiny->plain_mod,
                INPUT_DIM,
                COMMON_DIM
            );

            #ifdef BERT_TIMING
            t_total_repacking += interval(t_repacking_2);
            #endif 

            if(party == ALICE){
                h4 = ss_to_he_server(
                    lin.he_8192_tiny, 
                    ln_output_col,
                    INPUT_DIM*COMMON_DIM);
            } else{
                ss_to_he_client(lin.he_8192_tiny, ln_output_col, ln_size);
            }


            delete[] ln_input_cross;
            delete[] ln_input_row;
            delete[] ln_output_row;
            delete[] ln_output_col;
            delete[] ln_weight;
            delete[] ln_bias;
        }
    

        // -------------------- Linear #3 -------------------- //
        {
            int gelu_input_size = 128*3072;
            int gelu_cts_size = gelu_input_size / lin.he_8192_tiny->poly_modulus_degree;
            uint64_t* gelu_input_cross =
                new uint64_t[gelu_input_size];
            uint64_t* gelu_input_col =
                new uint64_t[gelu_input_size];
            uint64_t* gelu_output_col =
                new uint64_t[gelu_input_size];

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #3 HE" << endl;
                #ifdef BERT_TIMING
                auto t_linear3 = high_resolution_clock::now();
                #endif 
                vector<Ciphertext> h5 = lin.linear_2(
                lin.he_8192_tiny,
                h4, 
                lin.pp_3[layer_id],
                lin.data_lin3
                );
                #ifdef BERT_TIMING
                t_total_linear3 += interval(t_linear3);
                #endif 
                cout << "-> Layer - " << layer_id << ": Linear #3 HE done " << endl;
                he_to_ss_server(lin.he_8192_tiny, h5, gelu_input_cross);
            } else{
                he_to_ss_client(lin.he_8192_tiny, gelu_input_cross, gelu_cts_size, lin.data_lin3);
            }

            #ifdef BERT_TIMING
            auto t_repacking = high_resolution_clock::now();
            #endif 


            lin.plain_col_packing_postprocess(
                gelu_input_cross,
                gelu_input_col,
                true,
                lin.data_lin3
            );

            #ifdef BERT_TIMING
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
            #endif 

            // mod p
            nl.gt_p_sub(
                NL_NTHREADS,
                gelu_input_col,
                lin.he_8192_tiny->plain_mod,
                gelu_input_col,
                gelu_input_size,
                NL_ELL,
                11,
                11
            );

            #ifdef BERT_TIMING
            t_total_gt_sub += interval(t_gt_sub);
            auto t_gelu = high_resolution_clock::now();
            #endif 

            nl.gelu(
                NL_NTHREADS,
                gelu_input_col,
                gelu_output_col,
                gelu_input_size,
                NL_ELL,
                11
            );

            #ifdef BERT_TIMING
            t_total_gelu += interval(t_gelu);
            auto t_shift = high_resolution_clock::now();
            #endif 

            nl.right_shift(
                NL_NTHREADS,
                gelu_output_col,
                11 - 4,
                gelu_output_col,
                gelu_input_size,
                64,
                11
            );

            #ifdef BERT_TIMING
            t_total_shift += interval(t_shift);
            #endif 

            // FixArray tmp = nl.to_public(gelu_output_col, 128*3072, 64, 4);
            // save_to_file(tmp.data, 128, 3072, "./inter_result/linear4_input.txt");

            // return 0;

            if(party == ALICE){
                h6 = ss_to_he_server(
                    lin.he_8192_tiny, 
                    gelu_output_col,
                    gelu_input_size);
                
                // send_encrypted_vector(io, h6);
            } else{
                ss_to_he_client(
                    lin.he_8192_tiny, 
                    gelu_output_col, 
                    gelu_input_size);
                // vector<Ciphertext> tmp(1);
                // recv_encrypted_vector(lin.he_8192_tiny->context, io, tmp);
                // print_ct(lin.he_8192_tiny, tmp[0], 8192);
            }

            delete[] gelu_input_cross;
            delete[] gelu_input_col;
            delete[] gelu_output_col;
        }

        {
            int ln_2_input_size = INPUT_DIM*COMMON_DIM;
            int ln_2_cts_size = ln_2_input_size/lin.he_8192_tiny->poly_modulus_degree;

            uint64_t* ln_2_input_cross =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_input_row =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_output_row =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_output_col =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_weight_2 = new uint64_t[ln_2_input_size];
            uint64_t* ln_bias_2 = new uint64_t[ln_2_input_size];

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #4 HE " << endl;
                #ifdef BERT_TIMING
                auto t_linear4 = high_resolution_clock::now();
                #endif 
                vector<Ciphertext> h7 = lin.linear_2(
                    lin.he_8192_tiny,
                    h6, 
                    lin.pp_4[layer_id],
                    lin.data_lin4
                );
                #ifdef BERT_TIMING
                t_total_linear4 += interval(t_linear4);
                #endif 
                cout << "-> Layer - " << layer_id << ": Linear #4 HE done" << endl;
                he_to_ss_server(lin.he_8192_tiny, h7, ln_2_input_cross);
                ln_share_server(
                    layer_id,
                    lin.w_ln_2[layer_id],
                    lin.b_ln_2[layer_id],
                    ln_weight_2,
                    ln_bias_2
                );
            } else{
                he_to_ss_client(lin.he_8192_tiny, ln_2_input_cross, ln_2_cts_size, lin.data_lin4);
                ln_share_client(
                    ln_weight_2,
                    ln_bias_2
                );
            }

            #ifdef BERT_TIMING
            auto t_repacking = high_resolution_clock::now();
            #endif 
            // Post Processing
            lin.plain_col_packing_postprocess(
                ln_2_input_cross,
                ln_2_input_row,
                false,
                lin.data_lin4
            );

            #ifdef BERT_TIMING
            t_total_repacking += interval(t_repacking);
            auto t_gt_sub = high_resolution_clock::now();
            #endif 

            // mod p
            if(layer_id == 9 || layer_id == 10){
                nl.gt_p_sub(
                    NL_NTHREADS,
                    ln_2_input_row,
                    lin.he_8192_tiny->plain_mod,
                    ln_2_input_row,
                    ln_2_input_size,
                    NL_ELL,
                    8,
                    NL_SCALE
                );
            } else{
                nl.gt_p_sub(
                    NL_NTHREADS,
                    ln_2_input_row,
                    lin.he_8192_tiny->plain_mod,
                    ln_2_input_row,
                    ln_2_input_size,
                    NL_ELL,
                    9,
                    NL_SCALE
                );
            }

            #ifdef BERT_TIMING
            t_total_gt_sub += interval(t_gt_sub);
            auto t_ln = high_resolution_clock::now();
            #endif 

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

            #ifdef BERT_TIMING
            t_total_ln_2 += interval(t_ln);
            auto t_shift = high_resolution_clock::now();
            #endif 

            // update H1
            memcpy(h1_cache_12, ln_2_output_row, ln_2_input_size*sizeof(uint64_t));

            // Rescale
            nl.right_shift(
                NL_NTHREADS,
                ln_2_output_row,
                12 - 5,
                ln_2_output_row,
                ln_2_input_size,
                64,
                NL_SCALE
            );

            #ifdef BERT_TIMING
            t_total_shift += interval(t_shift);
            auto t_repacking_2 = high_resolution_clock::now();
            #endif 

            lin.plain_col_packing_preprocess(
                ln_2_output_row,
                ln_2_output_col,
                lin.he_8192_tiny->plain_mod,
                INPUT_DIM,
                COMMON_DIM
            );

            #ifdef BERT_TIMING
            t_total_repacking += interval(t_repacking_2);
            #endif 

            if(layer_id == 11){
                // Using Scale of 12 as 
                memcpy(h98, h1_cache_12, COMMON_DIM*sizeof(uint64_t));
            } else{
                if(party == ALICE){
                    h1 = ss_to_he_server(
                    lin.he_8192, 
                    ln_2_output_col,
                    ln_2_input_size);
                } else{
                    ss_to_he_client(
                        lin.he_8192, 
                        ln_2_output_col, 
                        ln_2_input_size);
                }
            }

            delete[] ln_2_input_cross;
            delete[] ln_2_input_row;
            delete[] ln_2_output_row;
            delete[] ln_2_output_col;
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

    #ifdef BERT_TIMING
    auto t_pc = high_resolution_clock::now();
    #endif 

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
        NL_SCALE
    );

    for(int i = 0; i < NUM_CLASS; i++){
        h99[i] += bp[i];
    }

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
        NL_SCALE
    );

    for(int i = 0; i < NUM_CLASS; i++){
        h101[i] += bc[i];
    }

    #ifdef BERT_TIMING
    cout << "> [TIMING]: linear1 takes " << t_total_linear1 << " sec" << endl;
    cout << "> [TIMING]: linear2 takes " << t_total_linear2 << " sec" << endl;
    cout << "> [TIMING]: linear3 takes " << t_total_linear3 << " sec" << endl;
    cout << "> [TIMING]: linear4 takes " << t_total_linear4 << " sec" << endl;

    cout << "> [TIMING]: softmax takes " << t_total_softmax << " sec" << endl;
    cout << "> [TIMING]: mul v takes " << t_total_mul_v << " sec" << endl;
    cout << "> [TIMING]: gelu takes " << t_total_gelu << " sec" << endl;
    cout << "> [TIMING]: ln_1 takes " << t_total_ln_1 << " sec" << endl;
    cout << "> [TIMING]: ln_2 takes " << t_total_ln_2 << " sec" << endl;

    cout << "> [TIMING]: repacking takes " << t_total_repacking << " sec" << endl;
    cout << "> [TIMING]: gt_sub takes " << t_total_gt_sub << " sec" << endl;
    cout << "> [TIMING]: shift takes " << t_total_shift << " sec" << endl;

    cout << "> [TIMING]: conversion takes " << t_total_conversion << " sec" << endl;
    cout << "> [TIMING]: ln_share takes " << t_total_ln_share << " sec" << endl;


    cout << "> [TIMING]: " << interval(t_pc) << " sec" << endl; 
    #endif 

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