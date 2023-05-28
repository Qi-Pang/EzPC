#ifndef BERT_H__
#define BERT_H__

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include "linear.h"
#include "nonlinear.h"

#define NL_NTHREADS 12

#define NL_ELL 37
#define NL_SCALE 12

#define ATTENTION_LAYERS 1

#define BERT_DEBUG

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

    Bert(int party, int port, string address);
    ~Bert();


    // void he_to_ss_server(HE* he, vector<Ciphertext> in, uint64_t* output);
    // vector<Ciphertext> ss_to_he_server(HE* he, uint64_t* input);

    // void he_to_ss_client(HE* he, uint64_t* output);
    // void ss_to_he_client(HE* he, uint64_t* input);

    void run_server();

    void run_client();
	
};

#endif