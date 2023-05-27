#ifndef BERT_H__
#define BERT_H__

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include "linear.h"
#include "nonlinear.h"

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

    Bert(int party, int port, string address);
    ~Bert();

    void run();
	
};

#endif