/* reads .npy files and convert them to tensorflow::Tensor */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <google/protobuf/repeated_field.h>
#include "tensorflow/compiler/aot/ex/test_graph.h"
#include "tensorflow/compiler/aot/ex/common_tensor.pb.h"

using namespace std;
using namespace google::protobuf;

void list_common_tensors(const CommonTensors & common_tensors) {
    for(int i=0; i<common_tensors.data_size(); i++) {
        const CommonTensor& common_tensor = common_tensors.data(i);
        for(int j=0; j<common_tensor.data_size(); j++)
            cout << common_tensor.data(j) << endl;
    }
}

std::vector<float> to_vector(const CommonTensor& tensor) {
    const RepeatedField<float>& dat = tensor.data();
    std::vector<float> ret;
    ret.resize(dat.size());
    std::copy(dat.begin(), dat.end(), ret.begin());
    return ret;
}

void test_graph(const std::vector<float> in_1, const std::vector<float> in_2) {
    MatMulAndAddComp compute;

    std::copy(in_1.begin(), in_1.end(), compute.arg0_data());
    std::copy(in_2.begin(), in_2.end(), compute.arg1_data());
    
    compute.Run();

    const MatMulAndAddComp& compute_const = compute;
    
    for(int i=0; i<4;i++) {
        cout << compute_const.arg0(i/2, i%2) << " " << compute_const.arg1(i/2, i%2) << endl;
    }
    for(int i=0; i<4;i++) {
        cout << compute_const.result0(i/2, i%2) << endl;
    }
}

int main() {
    CommonTensors tensors;
    fstream in_file;
    in_file.open("./save/common_tensors.pb", ios::in | ios::binary);
    tensors.ParseFromIstream(&in_file);
    
    //list_common_tensors(tensors);
    
    std::vector<float> arg1, arg2;
    arg1 = to_vector(tensors.data(0));
    arg2 = to_vector(tensors.data(1));
    
    test_graph(arg1, arg2);
    
    return 0;
}