/*
Authors: Jinhao Zhu
*/

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include <vector>
#include <sstream>

using namespace std;

vector<vector<uint64_t>> read_data(const string& filename) {
    ifstream input_file(filename);
    vector<vector<uint64_t>> data;

    if (!input_file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    while (getline(input_file, line)) {
        vector<uint64_t> row;
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            row.push_back(stoll(cell));
        }

        data.push_back(row);
    }

    input_file.close();
    return data;
}

