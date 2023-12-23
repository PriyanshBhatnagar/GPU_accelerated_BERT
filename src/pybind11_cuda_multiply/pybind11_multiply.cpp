#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


extern void cu_multiply(float* A, float* B, float* C, int M, int N, int K, int X);

namespace py = pybind11;


py::array_t<float> mm_wrapper(py::array_t<float> a1, py::array_t<float> a2)
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();

	
	int X = 20;
	int N = 128;
	int K = 768;
	int M = 768;

	auto result = py::array(py::buffer_info(
		nullptr,           
		sizeof(float),    
		py::format_descriptor<float>::value, 
		buf1.ndim,          
		{ X, N * M }, 
		{ sizeof(float) * N * M, sizeof(float) } 
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* B = (float*)buf2.ptr;
	float* C = (float*)buf3.ptr;

	cu_multiply(A, B, C, M, N, K, X);

	return result;
}

PYBIND11_MODULE(cu_multiply_matrix, m) {
	m.def("multiply", &mm_wrapper, "Linear layer");

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}