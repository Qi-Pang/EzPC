#include "nonlinear.h"

NonLinear::NonLinear(int party, string address, int port){
    this->party = party;
    this->address = address;
    this->port = port;

    for(int i = 0; i < MAX_THREADS; ++i){
        iopackArr[i] = new IOPack(party, port + i, address);
        if (i & 1) {
            otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
            fpmath[i] = new FPMath(3 - party, iopackArr[i], otpackArr[i]);
        } else {
            otpackArr[i] = new OTPack(iopackArr[i], party);
            fpmath[i] = new FPMath(party, iopackArr[i], otpackArr[i]);
        }
    }
}

NonLinear::~NonLinear(){
    delete[] fpmath;
    delete[] iopackArr;
    delete[] otpackArr;   
}

vector<FixArray> NonLinear::softmax(vector<FixArray> input, int nthreads){
    std::thread operation_threads[nthreads];
    vector<vector<FixArray>> output_tmp;
    vector<FixArray> output;
    int dim = input.size();
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i){
        int offset = i * chunk_size;
        int nops;
        if (i == (nthreads - 1)) {
            nops = dim - offset;
        } else {
            nops = chunk_size;
        }
        vector<FixArray> in = {input.begin() + offset, input.begin() + offset + nops};
        vector<FixArray> out(nops);
        output_tmp.push_back(out);
        operation_threads[i] = thread(
            non_linear_thread_vector, i, &in, &output_tmp[i], FPMath::softmax_fix
        );
    }
    for (int i = 0; i < nthreads; ++i) {
        operation_threads[i].join();
        output.insert(output.end(), output_tmp[i].begin(), output_tmp[i].end());
    }
    return output;
}

vector<FixArray> NonLinear::softmax_iron(vector<FixArray> input, int nthreads){
    return {};
}

vector<FixArray> NonLinear::layer_norm(vector<FixArray> input, int nthreads){
    return {};
}

vector<FixArray> NonLinear::layer_norm_iron(vector<FixArray> input, int nthreads){
    return layer_norm(input, nthreads);
}

FixArray NonLinear::gelu(FixArray input, int nthreads){
    std::thread operation_threads[nthreads];
    int n = input.size;
    int s = input.s;
    int ell = input.ell;
    int signed_ = input.signed_;

    FixArray output(party, n, signed_, ell, s);

    int chunk_size = n / nthreads;
    for (int i = 0; i < nthreads; ++i){
        int offset = i * chunk_size;
        int nops;
        if (i == (nthreads - 1)) {
            nops = n - offset;
        } else {
            nops = chunk_size;
        }

        operation_threads[i] = thread(
            non_linear_thread, i, &input.data[offset], &output.data[offset], nops, FPMath::gelu_approx
        );
    }
    for (int i = 0; i < nthreads; ++i) {
        operation_threads[i].join();
    }
    return output;
}

FixArray NonLinear::gelu_iron(FixArray input, int nthreads){
    std::thread operation_threads[nthreads];
    int n = input.size;
    int s = input.s;
    int ell = input.ell;
    int signed_ = input.signed_;

    FixArray output(party, n, signed_, ell, s);

    int chunk_size = n / nthreads;
    for (int i = 0; i < nthreads; ++i){
        int offset = i * chunk_size;
        int nops;
        if (i == (nthreads - 1)) {
            nops = n - offset;
        } else {
            nops = chunk_size;
        }

        operation_threads[i] = thread(
            non_linear_thread, i, &input.data[offset], &output.data[offset], nops, FPMath::gelu_iron
        );
    }
    for (int i = 0; i < nthreads; ++i) {
        operation_threads[i].join();
    }
    return output;
}

FixArray NonLinear::tanh(FixArray input, int nthreads){
    std::thread operation_threads[nthreads];
    int n = input.size;
    int s = input.s;
    int ell = input.ell;
    int signed_ = input.signed_;

    FixArray output(party, n, signed_, ell, s);

    int chunk_size = n / nthreads;
    for (int i = 0; i < nthreads; ++i){
        int offset = i * chunk_size;
        int nops;
        if (i == (nthreads - 1)) {
            nops = n - offset;
        } else {
            nops = chunk_size;
        }

        operation_threads[i] = thread(
            non_linear_thread, i, &input.data[offset], &output.data[offset], nops, FPMath::tanh_approx
        );
    }
    for (int i = 0; i < nthreads; ++i) {
        operation_threads[i].join();
    }
    return output;
}

FixArray NonLinear::tanh_iron(FixArray input, int nthreads){
    std::thread operation_threads[nthreads];
    int n = input.size;
    int s = input.s;
    int ell = input.ell;
    int signed_ = input.signed_;

    FixArray output(party, n, signed_, ell, s);

    int chunk_size = n / nthreads;
    for (int i = 0; i < nthreads; ++i){
        int offset = i * chunk_size;
        int nops;
        if (i == (nthreads - 1)) {
            nops = n - offset;
        } else {
            nops = chunk_size;
        }

        operation_threads[i] = thread(
            non_linear_thread, i, &input.data[offset], &output.data[offset], nops, FPMath::tanh_approx
        );
    }
    for (int i = 0; i < nthreads; ++i) {
        operation_threads[i].join();
    }
    return output;
}

void NonLinear::non_linear_thread_vector(int tid, vector<FixArray>& input, vector<FixArray>& output, vector<FixArray> (FPMath::*func)(const vector<FixArray>& x)){
    output = (fpmath[tid]->*func)(input);
}
