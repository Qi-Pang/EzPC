#ifndef BERT_H__
#define BERT_H__

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include "linear.h"
#include "nonlinear.h"
#include "bert_utils.h"

#define NL_NTHREADS 12
#define NL_ELL 37
#define NL_SCALE 12
#define ATTENTION_LAYERS 12

#define NUM_CLASS 2

#define BERT_DEBUG
#define BERT_TIMING

using namespace std;

class Bert
{
public:
    int party;
    string address;
    int port;

    NetIO *io;

    Linear lin;
    NonLinear nl;

    FCMetadata data_lin1;
    FCMetadata data_lin2;
    FCMetadata data_lin3;
    FCMetadata data_lin4;

    struct BertModel bm;

    Bert(int party, int port, string address);
    ~Bert();


    void he_to_ss_server(HE* he, vector<Ciphertext> in, uint64_t* output);
    vector<Ciphertext> ss_to_he_server(HE* he, uint64_t* input, int length);

    void he_to_ss_client(HE* he, uint64_t* output, int length, const FCMetadata &data);
    void ss_to_he_client(HE* he, uint64_t* input, int length);

    void pc_bw_share_server(
        struct BertModel &bm,
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

    void run_server();

    int run_client(string input_fname);

    int run(string model_dir, string input_fname, string mask_fname);
	
};

#endif