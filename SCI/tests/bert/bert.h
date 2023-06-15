#ifndef BERT_H__
#define BERT_H__

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include "linear.h"
#include "nonlinear.h"
#include "matrix_utils.h"

#define NL_NTHREADS 12
#define NL_ELL 37
#define NL_SCALE 12


#define NUM_CLASS 2

#define BERT_DEBUG
// #define BERT_TIMING

using namespace std;

class Bert
{
public:
    int party;
    string address;
    int port;
    int conv_err;

    NetIO *io;

    Linear lin;
    NonLinear nl;

    BertModel bm;

    Bert(int party, int port, string address, string model_path);
    ~Bert();


    void he_to_ss_server(HE* he, vector<Ciphertext> in, uint64_t* output);
    vector<Ciphertext> ss_to_he_server(HE* he, uint64_t* input, int length);

    void he_to_ss_client(HE* he, uint64_t* output, int length, const FCMetadata &data);
    void ss_to_he_client(HE* he, uint64_t* input, int length);

    void he_to_ss_server_plain(HE* he, uint64_t* input, uint64_t* output, int length);
    void he_to_ss_client_plain(HE* he, uint64_t* output, int length);

    void ss_to_he_server_plain(HE* he, uint64_t* input, int length, uint64_t* output);
    void ss_to_he_client_plain(HE* he, uint64_t* input, int length);

    void pc_bw_share_server(
        uint64_t* wp,
        uint64_t* bp,
        uint64_t* wc,
        uint64_t* bc
        );
    void pc_bw_share_client(
        uint64_t* wp,
        uint64_t* bp,
        uint64_t* wc,
        uint64_t* bc
    );

    void ln_share_server(
        int layer_id,
        vector<uint64_t> &wln_input,
        vector<uint64_t> &bln_input,
        uint64_t* wln,
        uint64_t* bln
    );

    void ln_share_client(
        uint64_t* wln,
        uint64_t* bln
    );

    vector<double> run(string input_fname, string mask_fname);
	
};

#endif