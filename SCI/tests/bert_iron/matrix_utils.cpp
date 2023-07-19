#include "matrix_utils.h"

void save_to_file_2(uint64_t* matrix, size_t rows, size_t cols, const char* filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file << (int64_t)matrix[i * cols + j];
            if (j != cols - 1) {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}

void mat_mul(const vector<vector<uint64_t>>& matrix1, const vector<vector<uint64_t>>& matrix2, uint64_t* result, size_t rows1, size_t cols1, size_t cols2) {
    // Perform matrix multiplication
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows1; ++i) {
        for (size_t j = 0; j < cols2; ++j) {
            uint64_t sum = 0;
            for (size_t k = 0; k < cols1; ++k) {
                // Access the elements in row-major order
                uint64_t element1 = matrix1[i][k];
                uint64_t element2 = matrix2[k][j];

                // Perform the multiplication and accumulate the sum
                sum += element1 * element2;
            }
            result[i * cols2 + j] = sum;
        }
    }
}

void mat_mul_ptr(const uint64_t* matrix1, const uint64_t* matrix2, uint64_t* result, size_t rows1, size_t cols1, size_t cols2) {
    // Perform matrix multiplication
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows1; ++i) {
        for (size_t j = 0; j < cols2; ++j) {
            uint64_t sum = 0;
            for (size_t k = 0; k < cols1; ++k) {
                // Access the elements in row-major order
                uint64_t element1 = matrix1[i * cols1 + k];
                uint64_t element2 = matrix2[k * cols2 + j];

                // Perform the multiplication and accumulate the sum
                sum += element1 * element2;
            }
            result[i * cols2 + j] = sum;
        }
    }
}

void transpose(const uint64_t* inputMatrix, uint64_t* outputMatrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            outputMatrix[j * rows + i] = inputMatrix[i * cols + j];
        }
    }
}

void linear_1_plain(
    uint64_t* input,
    vector<vector<vector<uint64_t>>> w_q,
    vector<vector<vector<uint64_t>>> w_k,
    vector<vector<vector<uint64_t>>> w_v,
    vector<vector<uint64_t>> b_q,
    vector<vector<uint64_t>> b_k,
    vector<vector<uint64_t>> b_v,
    uint64_t* q,
    uint64_t* k,
    uint64_t* v
){
    vector<vector<uint64_t>> input_vec;
    for(int i =0; i < 128; i++){
        vector<uint64_t> tmp(768);
        for(int j = 0; j < 768; j++){
            tmp[j] = input[i*768 + j];
        }
        input_vec.push_back(tmp);
    }
    for(int pack_id = 0; pack_id < 12; pack_id++){
        uint64_t tmp_q[128*64]; 
        uint64_t tmp_k[128*64];
        uint64_t tmp_k_trans[128*64];
        uint64_t tmp_v[128*64];
        uint64_t tmp_qk[128*128];

        mat_mul(
            input_vec,
            w_q[pack_id],
            tmp_q,
            128,
            768,
            64
        );


        mat_mul(
            input_vec,
            w_k[pack_id],
            tmp_k,
            128,
            768,
            64
        );

        mat_mul(
            input_vec,
            w_v[pack_id],
            tmp_v,
            128,
            768,
            64
        );

        for(int i =0; i < 128; i++){
            for(int j = 0; j < 64; j++){
                tmp_q[i*64 + j] += b_q[pack_id][j];
                tmp_k[i*64 + j] += b_k[pack_id][j];
                tmp_v[i*64 + j] += b_v[pack_id][j];
            }
        }

        transpose(tmp_k, tmp_k_trans, 128, 64);
        
        memcpy(&q[pack_id*128*64], tmp_q, 128*64*sizeof(uint64_t));
        memcpy(&k[pack_id*128*64], tmp_k_trans, 128*64*sizeof(uint64_t));
        memcpy(&v[pack_id*128*64], tmp_v, 128*64*sizeof(uint64_t));
    }
}

void linear_2_plain(
    uint64_t* input,
    vector<vector<uint64_t>> w,
    vector<uint64_t> b,
    int dim1,
    int dim2,
    int dim3,
    uint64_t* output
){
    uint64_t* tmp_out = new uint64_t[dim1*dim3];
    vector<vector<uint64_t>> input_vec;
    for(int i =0; i < dim1; i++){
        vector<uint64_t> tmp(dim2);
        for(int j = 0; j < dim2; j++){
            tmp[j] = input[i*dim2 + j];
        }
        input_vec.push_back(tmp);
    }

    mat_mul(
        input_vec,
        w,
        tmp_out,
        dim1,
        dim2,
        dim3
    );

    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim3; j++){
            output[i*dim3 + j] = tmp_out[i*dim3 + j] + b[j];
        }
    }
    delete[] tmp_out;
}