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

    

    Bert bt(party, port, address, "/home/ubuntu/iron/mrpc/weights_txt_right/");

    auto start = high_resolution_clock::now();
    
    vector<vector<double>> inference_results;
    vector<int> predicted_labels;

    if(party == ALICE){
        for(int i = 0; i < 1; i++ ){
            cout << "==>> Inference sample #" << i << endl;
            vector<double> result = bt.run("", "");
            if(i % 10 == 0){
                cout << "Conv Error: " << bt.conv_err << endl;
            }
        }
        cout << "Conv Error: " << bt.conv_err << endl;
    } else{
        ofstream file("/home/ubuntu/accuracy/EzPC/iron_test.txt");
        if (!file) {
            std::cerr << "Could not open the file!" << std::endl;
            return {};
        }
        for(int i = 0; i < 1; i++ ){
            cout << "==>> Inference sample #" << i << endl;
            vector<double> result = bt.run(
                "/home/ubuntu/iron/mrpc/weights_txt_right/inputs_" + to_string(i) + "_data.txt",
                "/home/ubuntu/iron/mrpc/weights_txt_right/inputs_" + to_string(i) +  "_mask.txt"
                );
            if(result.size() == 1){
                file << result[0]<< endl;
            } else{
                // inference_results.push_back(result);
                auto max_ele = max_element(result.begin(), result.end());
                int max_index = distance(result.begin(), max_ele);
                // predicted_labels.push_back(max_index);
                file << max_index << "," 
                        << result[0]<< "," 
                        << result[1] << endl;
            }
        }
        file.close();
    }
    
    // cout << "Prediction: " << result << endl;
    auto end = high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    
    cout << "-> End to end takes: " << interval.count() << "sec" << endl;

    return 0;
}
