#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__device__ inline int64_t StridedIndex(size_t linear_index, const CudaVec& shape,
                                      const CudaVec& strides, size_t offset) {
  int64_t index = static_cast<int64_t>(offset);
  for (uint32_t d = shape.size; d > 0; --d) {
    const uint32_t dim = d - 1;
    const int64_t coordinate = static_cast<int64_t>(linear_index % shape.data[dim]);
    index += coordinate * strides.data[dim];
    linear_index /= shape.data[dim];
  }
  return index;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    out[gid] = a[StridedIndex(gid, shape, strides, offset)];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}



__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size,
                                  CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[StridedIndex(gid, shape, strides, offset)] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  if (a.size == 0) return;
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                             VecToCuda(strides), offset);
  /// END SOLUTION
}



__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size,
                                   CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[StridedIndex(gid, shape, strides, offset)] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  if (size == 0) return;
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


#define DEFINE_CUDA_EWISE_BINARY(name, expression)                                \
  __global__ void name##Kernel(const scalar_t* a, const scalar_t* b,               \
                               scalar_t* out, size_t size) {                      \
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    if (gid < size) {                                                             \
      const scalar_t x = a[gid];                                                  \
      const scalar_t y = b[gid];                                                  \
      out[gid] = (expression);                                                    \
    }                                                                            \
  }                                                                              \
  void name(const CudaArray& a, const CudaArray& b, CudaArray* out) {               \
    if (out->size == 0) return;                                                   \
    CudaDims dim = CudaOneDim(out->size);                                          \
    name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);       \
  }

#define DEFINE_CUDA_SCALAR_BINARY(name, expression)                               \
  __global__ void name##Kernel(const scalar_t* a, scalar_t val,                    \
                               scalar_t* out, size_t size) {                      \
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    if (gid < size) {                                                             \
      const scalar_t x = a[gid];                                                  \
      out[gid] = (expression);                                                    \
    }                                                                            \
  }                                                                              \
  void name(const CudaArray& a, scalar_t val, CudaArray* out) {                     \
    if (out->size == 0) return;                                                   \
    CudaDims dim = CudaOneDim(out->size);                                          \
    name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);         \
  }

#define DEFINE_CUDA_EWISE_UNARY(name, expression)                                 \
  __global__ void name##Kernel(const scalar_t* a, scalar_t* out, size_t size) {     \
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    if (gid < size) {                                                             \
      const scalar_t x = a[gid];                                                  \
      out[gid] = (expression);                                                    \
    }                                                                            \
  }                                                                              \
  void name(const CudaArray& a, CudaArray* out) {                                   \
    if (out->size == 0) return;                                                   \
    CudaDims dim = CudaOneDim(out->size);                                          \
    name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);              \
  }

DEFINE_CUDA_EWISE_BINARY(EwiseMul, x * y)
DEFINE_CUDA_SCALAR_BINARY(ScalarMul, x * val)
DEFINE_CUDA_EWISE_BINARY(EwiseDiv, x / y)
DEFINE_CUDA_SCALAR_BINARY(ScalarDiv, x / val)
DEFINE_CUDA_SCALAR_BINARY(ScalarPower, powf(x, val))

DEFINE_CUDA_EWISE_BINARY(EwiseMaximum, x != x ? x : (x > y ? x : y))
DEFINE_CUDA_SCALAR_BINARY(ScalarMaximum, x != x ? x : (x > val ? x : val))
DEFINE_CUDA_EWISE_BINARY(EwiseEq, x == y ? 1.0f : 0.0f)
DEFINE_CUDA_SCALAR_BINARY(ScalarEq, x == val ? 1.0f : 0.0f)
DEFINE_CUDA_EWISE_BINARY(EwiseGe, x >= y ? 1.0f : 0.0f)
DEFINE_CUDA_SCALAR_BINARY(ScalarGe, x >= val ? 1.0f : 0.0f)

DEFINE_CUDA_EWISE_UNARY(EwiseLog, logf(x))
DEFINE_CUDA_EWISE_UNARY(EwiseExp, expf(x))
DEFINE_CUDA_EWISE_UNARY(EwiseTanh, tanhf(x))

#undef DEFINE_CUDA_EWISE_BINARY
#undef DEFINE_CUDA_SCALAR_BINARY
#undef DEFINE_CUDA_EWISE_UNARY

#define MATMUL_TILE 16

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b,
                             scalar_t* out, uint32_t M, uint32_t N,
                             uint32_t P) {
  __shared__ scalar_t As[2 * MATMUL_TILE][MATMUL_TILE];
  __shared__ scalar_t Bs[MATMUL_TILE][2 * MATMUL_TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * (2 * MATMUL_TILE) + ty;
  int col = blockIdx.x * (2 * MATMUL_TILE) + tx;

  scalar_t c00 = 0.0f;
  scalar_t c01 = 0.0f;
  scalar_t c10 = 0.0f;
  scalar_t c11 = 0.0f;

  for (int t = 0; t < N; t += MATMUL_TILE) {
    As[ty][tx] =
        (row < M && t + tx < N)
            ? a[row * N + t + tx] : 0.0f;

    As[ty + MATMUL_TILE][tx] =
        (row + MATMUL_TILE < M && t + tx < N)
            ? a[(row + MATMUL_TILE) * N + t + tx] : 0.0f;

    Bs[ty][tx] =
        (t + ty < N && col < P)
            ? b[(t + ty) * P + col] : 0.0f;

    Bs[ty][tx + MATMUL_TILE] =
        (t + ty < N && col + MATMUL_TILE < P)
            ? b[(t + ty) * P + col + MATMUL_TILE] : 0.0f;

    __syncthreads();

    for (int k = 0; k < MATMUL_TILE; ++k) {
      scalar_t a0 = As[ty][k];
      scalar_t a1 = As[ty + MATMUL_TILE][k];
      scalar_t b0 = Bs[k][tx];
      scalar_t b1 = Bs[k][tx + MATMUL_TILE];

      c00 += a0 * b0;
      c01 += a0 * b1;
      c10 += a1 * b0;
      c11 += a1 * b1;
    }

    __syncthreads();
  }

  if (row < M && col < P)
    out[row * P + col] = c00;

  if (row < M && col + MATMUL_TILE < P)
    out[row * P + col + MATMUL_TILE] = c01;

  if (row + MATMUL_TILE < M && col < P)
    out[(row + MATMUL_TILE) * P + col] = c10;

  if (row + MATMUL_TILE < M && col + MATMUL_TILE < P)
    out[(row + MATMUL_TILE) * P + col + MATMUL_TILE] = c11;
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  if (M == 0 || P == 0) return;
  dim3 block(MATMUL_TILE, MATMUL_TILE);
  dim3 grid((P + 2 * MATMUL_TILE - 1) / (2 * MATMUL_TILE),
            (M + 2 * MATMUL_TILE - 1) / (2 * MATMUL_TILE));
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////


__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out,
                                size_t out_size, size_t reduce_size) {
  size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;

  scalar_t max_val = -INFINITY;
  size_t start = gid * reduce_size;

  for (size_t i = 0; i < reduce_size; ++i) {
    scalar_t val = a[start + i];
    if (val > max_val || isnan(val)) max_val = val;
  }

  out[gid] = max_val;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  if (out->size == 0) return;
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}



__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out,
                                size_t out_size, size_t reduce_size) {
  size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;

  scalar_t sum = 0.0f;
  size_t start = gid * reduce_size;

  for (size_t i = 0; i < reduce_size; ++i) {
    sum += a[start + i];
  }

  out[gid] = sum;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  if (out->size == 0) return;
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
