#include "bert.h"

#ifdef BERT_PERF
double t_total_linear1 = 0;
double t_total_linear2 = 0;
double t_total_linear3 = 0;
double t_total_linear4 = 0;

double t_total_softmax = 0;
double t_total_mul_qk = 0;
double t_total_mul_v = 0;
double t_total_gelu = 0;
double t_total_ln_1 = 0;
double t_total_ln_2 = 0;

double t_total_preproc = 0;
double t_total_postproc = 0;

double t_total_conversion = 0;

double t_total_ln_share = 0;

double n_rounds = 0;
double n_comm = 0;
#endif 

inline double interval(chrono::_V2::system_clock::time_point start){
    auto end = high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    return interval.count();
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

        #ifdef BERT_PERF
        auto t_load_model = high_resolution_clock::now();
        #endif 

        struct BertModel bm = 
            load_model(model_path, NUM_CLASS);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Loading Model takes: " << interval(t_load_model) << "sec" << endl;
        auto t_model_preprocess = high_resolution_clock::now();
        #endif 

        lin.weights_preprocess(bm);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Model Preprocessing takes: " << interval(t_model_preprocess) << "sec" << endl;
        #endif 
    }
    cout << "> Bert intialized done!" << endl << endl;

}

Bert::~Bert() {
    
}

vector<Plaintext> Bert::he_to_ss_server(HE* he, vector<Ciphertext> in, const FCMetadata &data){
    
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    
    PRG128 prg;
    int dim = in.size();
    int slot_count = he->poly_modulus_degree;
    uint64_t *secret_share = new uint64_t[dim*slot_count];
	prg.random_data(secret_share, dim*slot_count*sizeof(uint64_t));
    uint64_t mask_x = (NL_ELL == 64 ? -1 : ((1ULL << NL_ELL) - 1));

    for(int i = 0; i < dim*slot_count; i++){
        secret_share[i] &= mask_x;
    }

    vector<Plaintext> enc_noise;
    if(data.filter_w == OUTPUT_DIM){
        enc_noise = lin.preprocess_noise_1(he, secret_share, data);
    } else{
        enc_noise = lin.preprocess_noise_2(he, secret_share, data);
    }
    

    vector<uint64_t> p_2(slot_count, he->plain_mod_2);
    Plaintext pt_p_2 = lin.encode_vector(he, p_2.data(), data);

    vector<Ciphertext> cts;
    for(int i = 0; i < dim; i++){

        Ciphertext ct; 
        he->evaluator->sub_plain(in[i], enc_noise[i], ct);
        // he->evaluator->add_plain_inplace(ct, pt_p_2);
        // print_pt(he, pt, 8192);
        cts.push_back(ct);
    }
    
    send_encrypted_vector(io, cts);

    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 

    return enc_noise;
}

vector<Ciphertext> Bert::ss_to_he_server(HE* he, uint64_t* input, const FCMetadata &data){

    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 

    vector<Plaintext> share_server = lin.preprocess_ptr_plaintext(he, input, data);
    vector<Ciphertext> share_client(share_server.size());
    recv_encrypted_vector(he->context, io, share_client);
    for(int i = 0; i < share_server.size(); i++){
        he->evaluator->add_plain_inplace(share_client[i], share_server[i]);
    }
    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 
    return share_client;
}

vector<Plaintext> Bert::he_to_ss_client(HE* he, int length){
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    vector<Ciphertext> cts(length);
    vector<Plaintext> pts;
    recv_encrypted_vector(he->context, io, cts);
    for(int i = 0; i < length; i++){
        Plaintext pt;
        he->decryptor->decrypt_keep_zero_coeff(cts[i], pt);
        pts.push_back(pt);
    }
    #ifdef BERT_PERF
    t_total_conversion += interval(t_conversion);
    #endif 
    return pts;
}

void Bert::ss_to_he_client(HE* he, uint64_t* input, const FCMetadata &data){
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    vector<Ciphertext> cts = lin.preprocess_ptr(he, input, data);
    send_encrypted_vector(io, cts);
    #ifdef BERT_PERF
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

    #ifdef BERT_PERF
    auto t_ln_share = high_resolution_clock::now();
    #endif 

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

    #ifdef BERT_PERF
    t_total_ln_share += interval(t_ln_share);
    #endif 
}

void Bert::ln_share_client(
    uint64_t* wln,
    uint64_t* bln
){
    #ifdef BERT_PERF
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

    #ifdef BERT_PERF
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

    PRG128 prg;
    prg.random_data(random_share, length * sizeof(uint64_t));

    for(int i = 0; i < length; i++){
        random_share[i] &= mask_x;
    }

    io->send_data(random_share, length*sizeof(uint64_t));


    // Write wp share
    int offset = 0;
    for(int i = 0; i < COMMON_DIM; i++){
        for(int j = 0; j < COMMON_DIM; j++){
            wp[i*COMMON_DIM + j] = (lin.w_p[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }

    // Write bp share
    for(int i = 0; i < COMMON_DIM; i++){
        bp[i] = (lin.b_p[i] - random_share[offset]) & mask_x;
        offset++;
    }

    // Write w_c share
    for(int i = 0; i < COMMON_DIM; i++){
        for(int j = 0; j < NUM_CLASS; j++){
            wc[i*NUM_CLASS + j] = (lin.w_c[i][j] - random_share[offset]) & mask_x;
            offset++;
        }
    }

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
    vector<Ciphertext> h1(lin.data_lin1.filter_h / lin.data_lin1.nw);
    vector<Ciphertext> h2;
    vector<Ciphertext> h4;
    vector<Ciphertext> h6;

    #ifdef BERT_PERF
    n_rounds += io->num_rounds;
    n_comm += io->counter;

    for(int i = 0; i < MAX_THREADS; i++){
        n_rounds += nl.iopackArr[i]->get_rounds();
        n_comm += nl.iopackArr[i]->get_comm();
    }
    #endif

    if(party == ALICE){
        // -------------------- Preparing -------------------- //
        // Receive cipher text input
        recv_encrypted_vector(lin.he_4096->context, io, h1);
        cout << "> Receive input cts from client " << endl;
    } else{
        cout << "> Loading inputs" << endl;
        vector<vector<uint64_t>> input_plain = read_data(input_fname);
        softmax_mask = read_bias(mask_fname, 128);

        // Column Packing
        vector<uint64_t> input_col(COMMON_DIM * INPUT_DIM);
        for (int j = 0; j < COMMON_DIM; j++){
            for (int i = 0; i < INPUT_DIM; i++){
                input_col[j*INPUT_DIM + i] = neg_mod((int64_t)input_plain[i][j], (int64_t)lin.he_4096->plain_mod);
                h1_cache_12[i*COMMON_DIM + j] = input_plain[i][j];
            }
        }

        // Send cipher text input
        vector<Ciphertext> h1_cts = 
            lin.preprocess_vec(lin.he_4096, input_col, lin.data_lin1);
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
            int qkv_size = 3*v_size;

            uint64_t* q_matrix_row = new uint64_t[v_size];
            uint64_t* k_matrix_row = new uint64_t[v_size];
            uint64_t* v_matrix_row = new uint64_t[v_size];

            uint64_t* k_trans_matrix_row = new uint64_t[v_size];

            uint64_t* softmax_input_row = new uint64_t[qk_size];
            uint64_t* softmax_output_row = new uint64_t[softmax_size];
            uint64_t* softmax_v_row = new uint64_t[att_size];
            uint64_t* h2_concate = new uint64_t[att_size];
            uint64_t* h2_col = new uint64_t[att_size];
            vector<Plaintext> pts;
                    
            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #1 HE" << endl;
                #ifdef BERT_PERF
                auto t_linear1 = high_resolution_clock::now();
                #endif 
                vector<Ciphertext> q_k_v = lin.linear_1(
                    lin.he_4096,
                    h1,
                    lin.pp_1[layer_id],
                    lin.data_lin1
                );
                #ifdef BERT_PERF
                t_total_linear1 += interval(t_linear1);
                #endif 
                cout << "-> Layer - " << layer_id << ": Linear #1 done HE" << endl;
                pts = he_to_ss_server(lin.he_4096, q_k_v, lin.data_lin1);
            } else{
                int cts_len = lin.data_lin1.filter_w / lin.data_lin1.kw * 3 * 12;
                pts = he_to_ss_client(lin.he_4096, cts_len);
            }

            #ifdef BERT_PERF
            auto t_preproc = high_resolution_clock::now();
            #endif 

            auto qkv = lin.pt_postprocess_1(
                lin.he_4096,
                pts,
                lin.data_lin1,
                false
            );

            for(int i = 0; i < PACKING_NUM; i++){
                memcpy(
                    &q_matrix_row[i*INPUT_DIM*OUTPUT_DIM], 
                    qkv[i][0].data(), 
                    INPUT_DIM*OUTPUT_DIM*sizeof(uint64_t));
                
                memcpy(
                    &k_matrix_row[i*INPUT_DIM*OUTPUT_DIM], 
                    qkv[i][1].data(), 
                    INPUT_DIM*OUTPUT_DIM*sizeof(uint64_t));
                
                memcpy(
                    &v_matrix_row[i*INPUT_DIM*OUTPUT_DIM], 
                    qkv[i][2].data(), 
                    INPUT_DIM*OUTPUT_DIM*sizeof(uint64_t));
                
                transpose(
                    &k_matrix_row[i*INPUT_DIM*OUTPUT_DIM],
                    &k_trans_matrix_row[i*INPUT_DIM*OUTPUT_DIM],
                    INPUT_DIM,
                    OUTPUT_DIM
                );
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

            #ifdef BERT_PERF
            t_total_preproc += interval(t_preproc);
            auto t_mul_qk = high_resolution_clock::now();
            #endif 

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

            #ifdef BERT_PERF
            t_total_mul_qk += interval(t_mul_qk);
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
            nl.softmax_iron(
                NL_NTHREADS,
                softmax_input_row,
                softmax_output_row,
                12*INPUT_DIM,
                INPUT_DIM,
                NL_ELL,
                NL_SCALE);

            #ifdef BERT_PERF
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
                NL_SCALE,
                NL_SCALE
            );

            lin.concat(softmax_v_row, h2_concate, 12, 128, 64); 

            #ifdef BERT_PERF
            t_total_mul_v += interval(t_mul_v);
            auto t_postproc = high_resolution_clock::now();
            #endif  

            lin.plain_col_packing_preprocess(
                h2_concate,
                h2_col,
                lin.he_4096->plain_mod,
                INPUT_DIM,
                COMMON_DIM
            );

            #ifdef BERT_PERF
            t_total_postproc += interval(t_postproc);
            #endif 

            if(party == ALICE){
                h2 = ss_to_he_server(
                    lin.he_4096, 
                    h2_col,
                    lin.data_lin2);
            } else{
                ss_to_he_client(lin.he_4096, h2_col, lin.data_lin2);
            }

            delete [] q_matrix_row;
            delete [] k_matrix_row;
            delete [] v_matrix_row;
            delete [] k_trans_matrix_row;
            delete [] softmax_input_row;
            delete [] softmax_output_row;
            delete [] softmax_v_row;
            delete [] h2_concate;
            delete [] h2_col;
        }

        // -------------------- Linear #2 -------------------- //
        {
            int ln_size = INPUT_DIM*COMMON_DIM;
            uint64_t* ln_input_row = new uint64_t[ln_size];
            uint64_t* ln_output_row = new uint64_t[ln_size];
            uint64_t* ln_output_col = new uint64_t[ln_size];
            uint64_t* ln_weight = new uint64_t[ln_size];
            uint64_t* ln_bias = new uint64_t[ln_size];
            vector<Plaintext> pts;
            
            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #2 HE" << endl;

                #ifdef BERT_PERF
                auto t_linear2 = high_resolution_clock::now();
                #endif 
                
                vector<Ciphertext> h3 = lin.linear_2(
                    lin.he_4096,
                    h2, 
                    lin.pp_2[layer_id],
                    lin.data_lin2
                );
                
                #ifdef BERT_PERF
                t_total_linear2 += interval(t_linear2);
                #endif 
                
                cout << "-> Layer - " << layer_id << ": Linear #2 HE done " << endl;
                pts = he_to_ss_server(lin.he_4096, h3, lin.data_lin2);
                ln_share_server(
                    layer_id,
                    lin.w_ln_1[layer_id],
                    lin.b_ln_1[layer_id],
                    ln_weight,
                    ln_bias
                );
            } else{
                int cts_len = lin.data_lin2.filter_w / lin.data_lin2.kw;
                pts = he_to_ss_client(lin.he_4096, cts_len);
                ln_share_client(
                    ln_weight,
                    ln_bias
                );
            }

            #ifdef BERT_PERF
            auto t_preproc = high_resolution_clock::now();
            #endif 

            lin.pt_postprocess_2(
                lin.he_4096,
                pts,
                ln_input_row,
                lin.data_lin2,
                false
            );

            nl.right_shift(
                NL_NTHREADS,
                ln_input_row,
                NL_SCALE,
                ln_input_row,
                ln_size,
                NL_ELL,
                2*NL_SCALE
            );

            #ifdef BERT_PERF
            t_total_preproc += interval(t_preproc);
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

            #ifdef BERT_PERF
            t_total_ln_1 += interval(t_ln_1);
            auto t_postproc = high_resolution_clock::now();
            #endif 

            memcpy(h4_cache_12, ln_output_row, ln_size*sizeof(uint64_t));

            lin.plain_col_packing_preprocess(
                ln_output_row,
                ln_output_col,
                lin.he_4096->plain_mod,
                INPUT_DIM,
                COMMON_DIM
            );

            #ifdef BERT_PERF
            t_total_postproc += interval(t_postproc);
            #endif 

            if(party == ALICE){
                h4 = ss_to_he_server(
                    lin.he_4096, 
                    ln_output_col,
                    lin.data_lin3);
            } else{
                ss_to_he_client(lin.he_4096, ln_output_col, lin.data_lin3);
            }

            delete[] ln_input_row;
            delete[] ln_output_row;
            delete[] ln_output_col;
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
            vector<Plaintext> pts;

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #3 HE" << endl;

                #ifdef BERT_PERF
                auto t_linear3 = high_resolution_clock::now();
                #endif 


                vector<Ciphertext> h5 = lin.linear_2(
                    lin.he_4096,
                    h4, 
                    lin.pp_3[layer_id],
                    lin.data_lin3
                );

                #ifdef BERT_PERF
                t_total_linear3 += interval(t_linear3);
                #endif 

                cout << "-> Layer - " << layer_id << ": Linear #3 HE done " << endl;
                pts = he_to_ss_server(lin.he_4096, h5, lin.data_lin3);
            } else{
                int cts_len = lin.data_lin3.filter_w / lin.data_lin3.kw;
                pts = he_to_ss_client(lin.he_4096, cts_len);
            }

            #ifdef BERT_PERF
            auto t_preproc = high_resolution_clock::now();
            #endif 

            lin.pt_postprocess_2(
                lin.he_4096,
                pts,
                gelu_input_col,
                lin.data_lin3,
                true
            );

            nl.right_shift(
                NL_NTHREADS,
                gelu_input_col,
                NL_SCALE,
                gelu_input_col,
                gelu_input_size,
                NL_ELL,
                2*NL_SCALE
            );

            #ifdef BERT_PERF
            t_total_preproc += interval(t_preproc);
            auto t_gelu= high_resolution_clock::now();
            #endif 

            nl.gelu_iron(
                NL_NTHREADS,
                gelu_input_col,
                gelu_output_col,
                gelu_input_size,
                NL_ELL,
                NL_SCALE
            );

            #ifdef BERT_PERF
            t_total_gelu += interval(t_gelu);
            #endif 

            if(party == ALICE){
                h6 = ss_to_he_server(
                    lin.he_4096, 
                    gelu_output_col,
                    lin.data_lin4);
            } else{
                ss_to_he_client(
                    lin.he_4096, 
                    gelu_output_col, 
                    lin.data_lin4);
            }

            delete[] gelu_input_col;
            delete[] gelu_output_col;
        }

        // -------------------- Linear #4 -------------------- //
        {
            int ln_2_input_size = INPUT_DIM*COMMON_DIM;

            uint64_t* ln_2_input_row =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_output_row =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_2_output_col =
                new uint64_t[ln_2_input_size];
            uint64_t* ln_weight_2 = new uint64_t[ln_2_input_size];
            uint64_t* ln_bias_2 = new uint64_t[ln_2_input_size];
            vector<Plaintext> pts;

            if(party == ALICE){
                cout << "-> Layer - " << layer_id << ": Linear #4 HE " << endl;
                
                #ifdef BERT_PERF
                auto t_linear4 = high_resolution_clock::now();
                #endif 
                
                vector<Ciphertext> h7 = lin.linear_2(
                    lin.he_4096,
                    h6, 
                    lin.pp_4[layer_id],
                    lin.data_lin4
                );

                #ifdef BERT_PERF
                t_total_linear4 += interval(t_linear4);
                #endif 


                cout << "-> Layer - " << layer_id << ": Linear #4 HE done" << endl;
                pts = he_to_ss_server(lin.he_4096, h7, lin.data_lin4);
                ln_share_server(
                    layer_id,
                    lin.w_ln_2[layer_id],
                    lin.b_ln_2[layer_id],
                    ln_weight_2,
                    ln_bias_2
                );
            } else{
                int cts_len = lin.data_lin4.filter_w / lin.data_lin4.kw;
                pts = he_to_ss_client(lin.he_4096, cts_len);
                ln_share_client(
                    ln_weight_2,
                    ln_bias_2
                );
            }

            #ifdef BERT_PERF
            auto t_preproc = high_resolution_clock::now();
            #endif 

            lin.pt_postprocess_2(
                lin.he_4096,
                pts,
                ln_2_input_row,
                lin.data_lin4,
                false
            );

            nl.right_shift(
                NL_NTHREADS,
                ln_2_input_row,
                NL_SCALE,
                ln_2_input_row,
                ln_2_input_size,
                NL_ELL,
                2*NL_SCALE
            );

            #ifdef BERT_PERF
            t_total_preproc += interval(t_preproc);
            auto t_ln_2 = high_resolution_clock::now();
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

            #ifdef BERT_PERF
            t_total_ln_2 += interval(t_ln_2);
            auto t_postproc = high_resolution_clock::now();
            #endif 


            // update H1
            memcpy(h1_cache_12, ln_2_output_row, ln_2_input_size*sizeof(uint64_t));

            lin.plain_col_packing_preprocess(
                ln_2_output_row,
                ln_2_output_col,
                lin.he_4096->plain_mod,
                INPUT_DIM,
                COMMON_DIM
            );

            #ifdef BERT_PERF
            t_total_postproc += interval(t_postproc);
            #endif 

            if(layer_id == 11){
                // Using Scale of 12 as 
                memcpy(h98, h1_cache_12, COMMON_DIM*sizeof(uint64_t));
            } else{
                if(party == ALICE){
                    h1 = ss_to_he_server(
                    lin.he_4096, 
                    ln_2_output_col,
                    lin.data_lin1);
                } else{
                    ss_to_he_client(
                        lin.he_4096, 
                        ln_2_output_col, 
                        lin.data_lin1);
                }
            }

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

    #ifdef BERT_PERF
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
        2*NL_SCALE
    );

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

    // -------------------- TANH -------------------- //
    nl.tanh_iron(
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

    #ifdef BERT_PERF
    cout << "> [TIMING]: linear1 takes " << t_total_linear1 << " sec" << endl;
    cout << "> [TIMING]: linear2 takes " << t_total_linear2 << " sec" << endl;
    cout << "> [TIMING]: linear3 takes " << t_total_linear3 << " sec" << endl;
    cout << "> [TIMING]: linear4 takes " << t_total_linear4 << " sec" << endl;

    cout << "> [TIMING]: mul qk takes " << t_total_mul_qk << " sec" << endl;
    cout << "> [TIMING]: softmax takes " << t_total_softmax << " sec" << endl;
    cout << "> [TIMING]: mul v takes " << t_total_mul_v << " sec" << endl;
    cout << "> [TIMING]: gelu takes " << t_total_gelu << " sec" << endl;
    cout << "> [TIMING]: ln_1 takes " << t_total_ln_1 << " sec" << endl;
    cout << "> [TIMING]: ln_2 takes " << t_total_ln_2 << " sec" << endl;

    cout << "> [TIMING]: preprocessing takes " << t_total_preproc << " sec" << endl;
    cout << "> [TIMING]: postprocessing takes " << t_total_postproc << " sec" << endl;

    cout << "> [TIMING]: conversion takes " << t_total_conversion << " sec" << endl;
    cout << "> [TIMING]: ln_share takes " << t_total_ln_share << " sec" << endl;


    cout << "> [TIMING]: pool/classification takes" << interval(t_pc) << " sec" << endl; 

    uint64_t total_rounds = io->num_rounds;
    uint64_t total_comm = io->counter;

    for(int i = 0; i < MAX_THREADS; i++){
        total_rounds += nl.iopackArr[i]->get_rounds();
        total_comm += nl.iopackArr[i]->get_comm();
    }

    cout << "> [NETWORK]: Communication rounds: " << total_rounds - n_rounds << endl; 
    cout << "> [NETWORK]: Communication overhead: " << total_comm - n_comm << " bytes" << endl; 
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