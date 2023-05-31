#include "bert_utils.h"

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

vector<uint64_t> read_bias(const string& filename, int output_dim) {
    ifstream input_file(filename);
    vector<uint64_t> data;

    if (!input_file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    vector<uint64_t> sub_mat;
    int row_counting = 0;
    while (getline(input_file, line)) {
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            sub_mat.push_back(stoll(cell));
        }
        row_counting++;
        if (row_counting == output_dim) {
            data = sub_mat;
            sub_mat.clear();
            row_counting -= output_dim;
        }
    }

    input_file.close();
    return data;
}

vector<vector<vector<uint64_t>>> read_qkv_weights(const string& filename) {
    ifstream input_file(filename);
    vector<vector<vector<uint64_t>>> data;

    if (!input_file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    vector<vector<uint64_t>> sub_mat;
    int row_counting = 0;
    while (getline(input_file, line)) {
        vector<uint64_t> row;
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            row.push_back(stoll(cell));
        }
        sub_mat.push_back(row);
        row_counting++;
        if (row_counting == 768) {
            data.push_back(sub_mat);
            sub_mat.clear();
            row_counting -= 768;
        }
    }

    input_file.close();
    return data;
}

vector<vector<uint64_t>> read_qkv_bias(const string& filename) {
    ifstream input_file(filename);
    vector<vector<uint64_t>> data;

    if (!input_file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    vector<uint64_t> sub_mat;
    int row_counting = 0;
    while (getline(input_file, line)) {
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            sub_mat.push_back(stoll(cell));
        }
        row_counting++;
        if (row_counting == 64) {
            data.push_back(sub_mat);
            sub_mat.clear();
            row_counting -= 64;
        }
    }

    input_file.close();
    return data;
}

