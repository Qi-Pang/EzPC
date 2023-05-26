#include "FloatingPoint/fp-math.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>

using namespace sci;
using namespace std;

#define MAX_THREADS 12

class NonLinear {
public:
    int party;
    int port;
    string address;

    IOPack *iopackArr[MAX_THREADS];
    OTPack *otpackArr[MAX_THREADS];
    FPMath *fpmath[MAX_THREADS];

    NonLinear(int party, string address, int port);
    ~NonLinear();

    vector<FixArray> softmax(vector<FixArray> input, int nthreads);
    vector<FixArray> softmax_iron(vector<FixArray> input, int nthreads);

    vector<FixArray> layer_norm(vector<FixArray> input, int nthreads);
    vector<FixArray> layer_norm_iron(vector<FixArray> input, int nthreads);

    FixArray gelu(FixArray input, int nthreads);
    FixArray gelu_iron(FixArray input, int nthreads);
    FixArray tanh(FixArray input, int nthreads);
    FixArray tanh_iron(FixArray input, int nthreads);

    void non_linear_thread_vector(int tid, vector<FixArray>& input, vector<FixArray>& output, vector<FixArray> (FPMath::*)(const vector<FixArray>& x));
    void non_linear_thread(int tid, uint64_t* input, uint64_t* output, int nops, FixArray (FPMath::*)(const FixArray& x));
};