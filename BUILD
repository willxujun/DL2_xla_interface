load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_proto_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_proto_library_py")

py_binary(
    name = "interface",
    srcs = ["interface.py"],
    deps = [
        "//tensorflow/compiler/tf2xla:tf2xla_proto_py"
    ]
)

py_binary(
    name = "make_graph",
    srcs = ["make_graph.py"],
        )

tf_library(
    name = "test_graph",
    config = "test_graph_tfmatmulandadd.config.pbtxt",
    cpp_class = "MatMulAndAddComp",
    graph = "test_graph_tfmatmulandadd.pb",
    tags = ["manual"],
)

cc_binary(
    name = "run_func",
    srcs = ["use_graph.cc",],
    deps = [
        ":test_graph",  # link in the generated object file
        "//third_party/eigen3",
    ],
    linkopts = [
          "-lpthread",
    ]
)

cc_library(
    name = "common_tensors_lib",
    srcs = ["common_tensor.pb.cc"],
    hdrs = ["common_tensor.pb.h"],
)

cc_binary(
    name = "interface_cc",
    srcs = ["interface.cc"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
        ":common_tensor_proto",
        ":test_graph"
    ],
    linkopts = ["-L/usr/local/lib", "-lprotobuf"]
)

xla_proto_library(
    name = "common_tensor_proto",
    srcs = ["common_tensor.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
    ],
    tags = ["manual"]
)

xla_proto_library_py(
    name = "common_tensor_proto_py",
    srcs = ["common_tensor.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
    ],
)