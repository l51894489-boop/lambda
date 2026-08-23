#pragma once
#include <iostream>
#include <fstream>
#include <vector>

inline cl::Program load_opencl_program(cl::Context context, cl::Device device) {
    std::ifstream cl_file("kernel.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, std::make_pair(cl_string.c_str(), cl_string.length()));
    cl::Program program(context, sources);

    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "OpenCL Build Error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }
    return program;
}