#include "nonlinear.h"

NonLinear::NonLinear(int party, string address, int port){
    this->party = party;
    this->address = address;
    this->port = port;

    for(int i = 0; i < MAX_THREADS; ++i){
        this->iopackArr[i] = new IOPack(party, port + i, address);
        if (i & 1) {
            this->otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
            this->fpmath[i] = new FPMath(3 - party, iopackArr[i], otpackArr[i]);
        } else {
            this->otpackArr[i] = new OTPack(iopackArr[i], party);
            this->fpmath[i] = new FPMath(party, iopackArr[i], otpackArr[i]);
        }
    }
}

NonLinear::NonLinear(){}

NonLinear::~NonLinear(){
  for (int i = 0; i < MAX_THREADS; i++) {
    // delete this->iopackArr[i];
    // delete this->otpackArr[i];
    // delete this->fpmath[i];
  }
}

void softmax_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int array_size, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  vector<FixArray> input_array;
  for(int i = 0; i < num_ops; i++){
    input_array.push_back(fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s));
  }
  vector<FixArray> output_array = fpmath->softmax_fix(input_array);
  for(int i = 0; i < num_ops; i++){
    memcpy(&y[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
  }
}

void NonLinear::softmax(int nthreads, uint64_t* input, uint64_t* output, int dim, int array_size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = dim - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                softmax_thread, 
                i, 
                party, 
                &input[offset*array_size], 
                &output[offset*array_size], 
                lnum_ops,
                array_size,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void softmax_irons_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int array_size, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  vector<FixArray> input_array;
  for(int i = 0; i < num_ops; i++){
    input_array.push_back(fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s));
  }
  vector<FixArray> output_array = fpmath->softmax_fix_iron_1(input_array);
  for(int i = 0; i < num_ops; i++){
    memcpy(&y[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
  }
}

void NonLinear::softmax_iron(int nthreads, uint64_t* input, uint64_t* output, int dim, int array_size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = dim - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                softmax_irons_thread, 
                i, 
                party, 
                &input[offset*array_size], 
                &output[offset*array_size], 
                lnum_ops,
                array_size,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void layer_norm_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int array_size, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  vector<FixArray> input_array;
  for(int i = 0; i < num_ops; i++){
    input_array.push_back(fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s));
  }
  vector<FixArray> output_array = fpmath->layer_norm_iron(input_array);
  for(int i = 0; i < num_ops; i++){
    memcpy(&y[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
  }
}

void NonLinear::layer_norm(int nthreads, uint64_t* input, uint64_t* output, int dim, int array_size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = dim - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                layer_norm_thread, 
                i, 
                party, 
                &input[offset*array_size], 
                &output[offset*array_size], 
                lnum_ops,
                array_size,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void gelu_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->gelu_approx(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::gelu(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gelu_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void gelu_iron_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->gelu_iron(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::gelu_iron(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                gelu_iron_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void tanh_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->tanh_approx(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::tanh(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                tanh_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}

void tanh_iron_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int ell, int s, FPMath *fpmath) {
  int this_party;
  if (tid & 1) {
    this_party = 3 - party;
  } else {
    this_party = party;
  }
  FixArray input = fpmath->fix->input(this_party, num_ops, x, true, ell, s);
  FixArray output = fpmath->tanh_iron(input);
  memcpy(y, output.data, num_ops*sizeof(uint64_t));
}

void NonLinear::tanh_iron(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s){
    std::thread threads[nthreads];
    int chunk_size = size / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
        lnum_ops = size - offset;
        } else {
        lnum_ops = chunk_size;
        }
        threads[i] =
            std::thread(
                tanh_iron_thread, 
                i, 
                party, 
                &input[offset], 
                &output[offset], 
                lnum_ops,
                ell,
                s,
                this->fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
}