#include "bert.h"


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
        bm = load_model(model_path, NUM_CLASS);
        // lin.weights_preprocess(bm);
    }
    cout << "> Bert intialized done!" << endl << endl;

}

Bert::~Bert() {
    
}

void Bert::he_to_ss_server(HE* he, vector<Ciphertext> in, uint64_t* output){
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
}

vector<Ciphertext> Bert::ss_to_he_server(HE* he, uint64_t* input, int length){
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
                input_row[i*COMMON_DIM + j] = input_plain[i][j];
                h1_cache_12[i*COMMON_DIM + j] = input_plain[i][j] << 7;
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
                    softmax_input_row,
                    v_matrix_row
                );
                cout << "-> Layer - " << layer_id << ": Linear #1 done HE" << endl;

            } else{
                for(int i = 0; i < qk_size; i++){
                    softmax_input_row[i] = 0;
                }
                for(int i = 0; i < v_size; i++){
                    v_matrix_row[i] = 0;
                }
            }

            
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

            if(party == ALICE){
                io->recv_data(h2, att_size*sizeof(uint64_t));
                for(int i = 0; i < att_size; i++){
                    h2[i] += h2_concate[i];
                }

            } else{
                io->send_data(h2_concate, att_size*sizeof(uint64_t));
            }
            delete [] v_matrix_row;
            delete [] softmax_input_row;
            delete [] softmax_output_row;
            delete [] softmax_v_row;
            delete [] h2_concate;
        }

        // -------------------- Linear #2 -------------------- //
        {
            int ln_size = INPUT_DIM*COMMON_DIM;
            int ln_cts_size = ln_size / lin.he_8192_tiny->poly_modulus_degree;
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

            // nl.print_ss(ln_output_row, 16, NL_ELL, NL_SCALE);
            // return {};

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


            // FixArray tmp = nl.to_public(ln_output_row, 128*768, 64, 5);
            // save_to_file(tmp.data, 128, 768, "./inter_result/linear3_input.txt");

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
            int gelu_cts_size = gelu_input_size / lin.he_8192_tiny->poly_modulus_degree;
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


            nl.gelu(
                NL_NTHREADS,
                gelu_input_col,
                gelu_output_col,
                gelu_input_size,
                NL_ELL,
                11
            );

            nl.right_shift(
                NL_NTHREADS,
                gelu_output_col,
                11 - 4,
                gelu_output_col,
                gelu_input_size,
                64,
                11
            );

            // FixArray tmp = nl.to_public(gelu_output_col, 128*3072, 64, 4);
            // save_to_file(tmp.data, 128, 3072, "./inter_result/linear4_input.txt");

            // return 0;

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
            int ln_2_cts_size = ln_2_input_size/lin.he_8192_tiny->poly_modulus_degree;

                new uint64_t[ln_2_input_size];
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