/*
*/

#include "bert.h"
#include <fstream>

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int port = 8000;
string address = "127.0.0.1";
int num_threads = 4;
int bitlength = 37;


int main(int argc, char **argv) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.parse(argc, argv);

    cout << ">>> Evaluating Bert" << endl;
    cout << "-> Role: " << party << endl;
    cout << "-> Address: " << address << endl;
    cout << "-> Port: " << port << endl;
    cout << "<<<" << endl << endl;

    

    Bert bt(party, port, address);

    auto start = high_resolution_clock::now();
    // if(party == ALICE){
    //     bt.run_server();
    // } else{
    //     bt.run_client("/home/ubuntu/mrpc/weights_txt/inputs_0.txt");
    // }
    bt.run("/home/ubuntu/mrpc/weights_txt/", "/home/ubuntu/mrpc/weights_txt/inputs_0.txt");
    auto end = high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    
    cout << "-> End to end takes: " << interval.count() << "sec" << endl;

    return 0;
}
