/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "Math/math-functions.h"
#include "BuildingBlocks/aux-protocols.h"
#include "FloatingPoint/fixed-point.h"
#include <fstream>
#include <iostream>
#include <thread>

using namespace sci;
using namespace std;



#define MAX_THREADS 4

int party, port = 32000;
int num_threads = 4;
string address = "127.0.0.1";

int dim = 128*3072;//1ULL << 16;

int bw_x = 37;
int bw_y = 37;
int s_x = 12;
int s_y = 12;
int radix_base = 4; 

uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

IOPack *iopackArr[MAX_THREADS];
OTPack *otpackArr[MAX_THREADS];

/************* Poly. Approx Parameters  ************/

uint64_t a_share = round((0.020848611754127593) * (1 << s_x));
uint64_t b_share =  round((0.18352506127082727) * (1 << s_x));    
uint64_t c_share =  round((0.5410550166368381) * (1 << s_x));    
uint64_t d_share =  round((0.03798164612714154) * (1 << s_x));    
uint64_t e_share =  round((0.001620808531841547) * (1 << s_x));   

/************* Poly. Approx with Preproc. Parameters  ************/

//P(x):= (z+x+(-873371867916189/35184372088832))*(z+(556832097825773/17592186044416))+(863965981841159/1099511627776)
//z(x):=x*(x+(-5518456147909361/1125899906842624))
uint64_t pre_a_share = round((-5518456147909361/1125899906842624) * (1 << s_x));
uint64_t pre_b_share =  round((-873371867916189/35184372088832) * (1 << s_x));    
uint64_t pre_c_share =  round((556832097825773/17592186044416) * (1 << s_x));    
uint64_t pre_d_share =  round((863965981841159/1099511627776) * (1 << s_x));     

/************* Poly. Approx with Preproc. Parameters  ************/
 
uint64_t f_share =  round((0.5) * (1 << s_x));

uint64_t threshold =  round((2.7) * (1 << s_x));
uint64_t enc_two =  round((2) * (1 << s_x)); 
uint64_t enc_one =  round((2) * (1 << s_x));    

double gelu(double x) {
  const double sqrtTwoOverPi = std::sqrt(2.0 / M_PI);
  return 0.5 * x * (1.0 + std::tanh(sqrtTwoOverPi * (x + 0.044715 * std::pow(x, 3))));
}

/********************************************/

uint64_t computeULPErr(double calc, double actual, int SCALE) {
  int64_t calc_fixed = (double(calc) * (1ULL << SCALE));
  int64_t actual_fixed = (double(actual) * (1ULL << SCALE));
  uint64_t ulp_err = (calc_fixed - actual_fixed) > 0
                         ? (calc_fixed - actual_fixed)
                         : (actual_fixed - calc_fixed);
  return ulp_err;
}

void ScalarAdd(int dim, uint64_t a, uint64_t* b, uint64_t* c){
                for(int i = 0;i < dim;i++)
                {
                  c[i] = ((party == ALICE)?a:0)+b[i];
                }
}

void ScalarSub(int dim, uint64_t a, uint64_t* b, uint64_t* c){
                for(int i = 0;i < dim;i++)
                {
                  c[i] = ((party == ALICE)?a:0)-b[i];
                }
}

void ScalarSub(int dim, uint64_t* a, uint64_t b, uint64_t* c){
                for(int i = 0;i < dim;i++)
                {
                  c[i] = a[i]-((party == ALICE)?b:0);
                }
}

void ShareAdd(int dim, uint64_t* a, uint64_t* b, uint64_t* c){
                for(int i = 0;i < dim;i++)
                {
                  c[i] = a[i]+b[i];
                }
}

void ShareSub(int dim, uint64_t* a, uint64_t* b, uint64_t* c){
                for(int i = 0;i < dim;i++)
                {
                  c[i] = a[i]-b[i];
                }
}


void ShareAddConst(int dim, uint64_t* a, uint64_t b, uint64_t* c){
                for(int i = 0;i < dim;i++)
                {
                  c[i] = a[i]+b;
                }
}


void ScalarMul(int dim, uint64_t a, uint64_t* b, uint64_t* c){//TODO: is a encoded or not? otherwise rescaling necessary?
                for(int i = 0;i < dim;i++)
                {
                  c[i] = a*b[i];
                }
}

//z(x):=x*(x+(-5518456147909361/1125899906842624))
void ourGeluZFunc(auto math,int tid, int num_ops,uint64_t *x, uint64_t *output, int bw_x, int s_x){
    uint64_t *tmp = new uint64_t[num_ops];
    uint64_t *z_share_tmp = new uint64_t[num_ops];
    ScalarAdd(num_ops,pre_a_share,x,tmp);
    
    math->mult->hadamard_product(num_ops,tmp, x, z_share_tmp, bw_x, bw_x, bw_x, true, true, MultMode::None);
    math->trunc->truncate_and_reduce(num_ops, z_share_tmp, output, s_x, bw_x);
}

//P(x):= (z+x+(-873371867916189/35184372088832))*(z+(556832097825773/17592186044416))+(863965981841159/1099511627776)
void ourGeluPFunc(auto math,int tid, int num_ops,uint64_t *x, uint64_t *output, int bw_x , int s_x){
    //pos_x = x * (1 - 2*MSB(x)) TODO abs
    uint8_t *msb = new uint8_t[num_ops];
    uint64_t *two_x = new uint64_t[num_ops];
    uint64_t *abs_x = new uint64_t[num_ops];
    uint64_t *two_msb_x = new uint64_t[num_ops];
    math->aux->MSB(x,msb,num_ops,bw_x);
    ScalarMul(num_ops,2,x,two_x);
    for (int i = 0; i < num_ops; i++) {
      two_msb_x[i] = (msb[i] * two_x[i]);//2x*MSB(x)
    }
    ShareSub(num_ops,x,two_msb_x,abs_x);
    
    uint64_t *z = new uint64_t[num_ops];
    ourGeluZFunc(math,tid,num_ops,x,z,bw_x,s_x);
    
    uint64_t *tmp_1 = new uint64_t[num_ops];
    uint64_t *first_factor = new uint64_t[num_ops];
    ScalarAdd(num_ops,pre_b_share,x,tmp_1);
    ShareAdd(num_ops,tmp_1,z,first_factor);

    uint64_t *second_factor = new uint64_t[num_ops];
    ScalarAdd(num_ops,pre_c_share,z,second_factor);
    
    uint64_t *sum = new uint64_t[num_ops];
    math->mult->hadamard_product(num_ops,first_factor, second_factor, sum, bw_x, bw_x, bw_x, true, true, MultMode::None);
    math->trunc->truncate_and_reduce(num_ops, sum, output, s_x, bw_x);
    
    ScalarAdd(num_ops,pre_d_share,sum,output);//end of P function
    
    //add 0.5x
    uint64_t *x_f_tmp = new uint64_t[num_ops];
    uint64_t *x_f_factor = new uint64_t[num_ops];
    ScalarMul(num_ops, f_share, x, x_f_tmp);
    math->trunc->truncate_and_reduce(num_ops, x_f_tmp, x_f_factor, s_x, bw_x);
    
    ShareAdd(num_ops,x_f_factor,output,output);
}

void naiveGeluApprox(auto math,int tid, int num_ops,uint64_t *x, uint64_t *y, int bw_x, int s_x) {

   
    //pos_x = x * (1 - 2*MSB(x)) TODO abs
    uint8_t *msb = new uint8_t[num_ops];
    uint64_t *two_x = new uint64_t[num_ops];
    uint64_t *abs_x = new uint64_t[num_ops];
    uint64_t *two_msb_x = new uint64_t[num_ops];
    math->aux->MSB(x,msb,num_ops,bw_x);
    ScalarMul(num_ops,2,x,two_x);
    for (int i = 0; i < num_ops; i++) {
      two_msb_x[i] = (msb[i] * two_x[i]);//2x*MSB(x)
    }
    ShareSub(num_ops,x,two_msb_x,abs_x);
    
    uint64_t *xsquare_tmp = new uint64_t[num_ops];
    uint64_t *xsquare = new uint64_t[num_ops];
    uint64_t *xcube_tmp = new uint64_t[num_ops];
    uint64_t *xcube = new uint64_t[num_ops];
    uint64_t *xfour_tmp = new uint64_t[num_ops];
    uint64_t *xfour = new uint64_t[num_ops];
  

    math->mult->hadamard_product(num_ops, abs_x, abs_x, xsquare_tmp, bw_x, bw_x, bw_x, true, true, MultMode::None);
    math->trunc->truncate_and_reduce(num_ops, xsquare_tmp, xsquare, s_x, bw_x);

    math->mult->hadamard_product(num_ops, abs_x, xsquare, xcube_tmp, bw_x, bw_x, bw_x, true, true, MultMode::None);
    math->trunc->truncate_and_reduce(num_ops, xcube_tmp, xcube, s_x, bw_x);
    
    math->mult->hadamard_product(num_ops, abs_x, xcube, xfour_tmp, bw_x, bw_x, bw_x, true, true, MultMode::None);
    math->trunc->truncate_and_reduce(num_ops, xfour_tmp, xfour, s_x, bw_x);

    
    uint64_t *xfour_a_tmp = new uint64_t[num_ops];
    uint64_t *xfour_a_factor = new uint64_t[num_ops];
    ScalarMul(num_ops, a_share, xfour, xfour_a_tmp);
    math->trunc->truncate_and_reduce(num_ops, xfour_a_tmp, xfour_a_factor, s_x, bw_x);
    
    uint64_t *xcube_b_tmp = new uint64_t[num_ops];
    uint64_t *xcube_b_factor = new uint64_t[num_ops];
    ScalarMul(num_ops, b_share, xcube, xcube_b_tmp);
    math->trunc->truncate_and_reduce(num_ops, xcube_b_tmp, xcube_b_factor, s_x, bw_x);
    
    uint64_t *xsquare_c_tmp = new uint64_t[num_ops];
    uint64_t *xsquare_c_factor = new uint64_t[num_ops];
    ScalarMul(num_ops, c_share, xsquare, xsquare_c_tmp);
    math->trunc->truncate_and_reduce(num_ops, xsquare_c_tmp, xsquare_c_factor, s_x, bw_x);
    
    uint64_t *x_d_tmp = new uint64_t[num_ops];
    uint64_t *x_d_factor = new uint64_t[num_ops];
    ScalarMul(num_ops, d_share, abs_x, x_d_tmp);
    math->trunc->truncate_and_reduce(num_ops, x_d_tmp, x_d_factor, s_x, bw_x);
    
    uint64_t *x_f_tmp = new uint64_t[num_ops];
    uint64_t *x_f_factor = new uint64_t[num_ops];
    ScalarMul(num_ops, f_share, x, x_f_tmp);
    math->trunc->truncate_and_reduce(num_ops, x_f_tmp, x_f_factor, s_x, bw_x);
    
    ShareAddConst(num_ops,xsquare_c_factor,e_share,y);
    ShareSub(num_ops,y,x_d_factor,y);
    ShareSub(num_ops,y,xcube_b_factor,y);
    ShareAdd(num_ops,y,xfour_a_factor,y);    
    ShareAdd(num_ops,y,x_f_factor,y);
}


void gelu_thread(int tid, uint64_t *x, uint64_t *y, int num_ops) {

   MathFunctions *math;
    if (tid & 1) {
      math = new MathFunctions(3 - party, iopackArr[tid], otpackArr[tid]);
    } else {
      math = new MathFunctions(party, iopackArr[tid], otpackArr[tid]);
    }
    
  AuxProtocols* aux = math->aux;
  
  //x<2.7
  uint64_t *sub_1 = new uint64_t[num_ops];
  uint8_t *lt_2 = new uint8_t[num_ops];
  ScalarSub(num_ops,x,threshold,sub_1);
  aux->MSB(sub_1, lt_2, num_ops, bw_x);
  
  //x>-2.7
  uint64_t *sub_2 = new uint64_t[num_ops];
  uint8_t *gt_neg2 = new uint8_t[num_ops];
  ScalarSub(num_ops, (-1)*threshold, x,sub_2);
  aux->MSB(sub_2, gt_neg2, num_ops, bw_x);
  
  //gelu_approx
  uint64_t *gelu_approx_res = new uint64_t[num_ops];
  ourGeluPFunc(math,tid,num_ops,x,gelu_approx_res,bw_x,s_x);
  //naiveGeluApprox(math,tid,num_ops,x,gelu_approx_res,bw_x,s_x);
  
  //MUX 1: x<2
  uint64_t *tmp_mux1 = new uint64_t[num_ops];
  uint64_t *mux1_res = new uint64_t[num_ops];
  ShareSub(num_ops,gelu_approx_res,x,tmp_mux1);
  aux->multiplexer(lt_2,tmp_mux1,mux1_res,num_ops,bw_x,bw_x);
  ShareAdd(num_ops,x,mux1_res,y);
  
  //MUX 1: x>-2
  uint64_t *tmp_mux2 = new uint64_t[num_ops];
  uint64_t *mux2_res = new uint64_t[num_ops];
  ScalarSub(num_ops,y,0,tmp_mux2);
  aux->multiplexer(gt_neg2,tmp_mux2,mux2_res,num_ops,bw_x,bw_x);
  ScalarAdd(num_ops,0,mux2_res,y);
  
  delete math;   
}

  

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of gelu operations");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  
  std::cout << "Dim: " << dim << std::endl;

  amap.parse(argc, argv);

  assert(num_threads <= MAX_THREADS);

  /********** Setup IO and Base OTs ***********/
  /********************************************/
  for (int i = 0; i < num_threads; i++) {
    iopackArr[i] = new IOPack(party, port + i, address);
    if (i & 1) {
      otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
    } else {
      otpackArr[i] = new OTPack(iopackArr[i], party);
    }
  }
  std::cout << "All Base OTs Done" << std::endl;

  /************ Generate Test Data ************/
  /********************************************/
  PRG128 prg;

  uint64_t *x = new uint64_t[dim];
  uint64_t *y = new uint64_t[dim];

  prg.random_data(x, dim * sizeof(uint64_t));

  for (int i = 0; i < dim; i++) {
    x[i] &= mask_x;
  }


  /************** Fork Threads ****************/
  /********************************************/
  uint64_t total_comm = 0;
  uint64_t thread_comm[num_threads];
  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm();
  }

  auto start = clock_start();
  std::thread gelu_threads[num_threads];
  int chunk_size = dim / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lnum_ops;
    if (i == (num_threads - 1)) {
      lnum_ops = dim - offset;
    } else {
      lnum_ops = chunk_size;
    }
    
    gelu_threads[i] = std::thread(gelu_thread, i, x + offset, y + offset, lnum_ops); 
  }
  for (int i = 0; i < num_threads; ++i) {
    gelu_threads[i].join();
  }
  long long t = time_from(start);

  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm() - thread_comm[i];
    total_comm += thread_comm[i];
  }

  /************** Verification ****************/
  /********************************************/
  if (party == ALICE) {
    iopackArr[0]->io->send_data(x, dim * sizeof(uint64_t));
    iopackArr[0]->io->send_data(y, dim * sizeof(uint64_t));
  } else { // party == BOB
    uint64_t *x0 = new uint64_t[dim];
    uint64_t *y0 = new uint64_t[dim];
    iopackArr[0]->io->recv_data(x0, dim * sizeof(uint64_t));
    iopackArr[0]->io->recv_data(y0, dim * sizeof(uint64_t));

    uint64_t total_err = 0;
    uint64_t max_ULP_err = 0;
    for (int i = 0; i < dim; i++) {
      double dbl_x = (signed_val(x0[i] + x[i], bw_x)) / double(1LL << s_x);
      double dbl_y = (signed_val(y0[i] + y[i], bw_y)) / double(1LL << s_y);
      // TODO GELU function for verification
      double gelu_x = gelu(dbl_x);
      uint64_t err = computeULPErr(dbl_y, gelu_x, s_y);
    //   cout << "ULP Error: " << dbl_x << "," << dbl_y << "," << gelu_x << ","
    //   << err << endl;
      total_err += err;
      max_ULP_err = std::max(max_ULP_err, err);
    }

    cerr << "Average ULP error: " << total_err / dim << endl;
    cerr << "Max ULP error: " << max_ULP_err << endl;
    cerr << "Number of tests: " << dim << endl;

    delete[] x0;
    delete[] y0;
  }

  cout << "Number of gelu/s:\t" << (double(dim) / t) * 1e6 << std::endl;
  cout << "Gelu Time\t" << t / (1000.0) << " ms" << endl;
  cout << "Gelu Bytes Sent\t" << total_comm << " bytes" << endl;


  /******************* Cleanup ****************/
  /********************************************/
  /*delete[] x;
  delete[] y;
  for (int i = 0; i < num_threads; i++) {
    delete iopackArr[i];
    delete otpackArr[i];
  }*/
}
