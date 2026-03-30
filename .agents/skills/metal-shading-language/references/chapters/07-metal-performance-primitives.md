## 7 Metal Performance Primitives

All OS: Metal 4 and later support Metal Performance Primitives.

Metal Performance Primitives is a library of optimized primitives that are designed to be efficient and performant on Apple silicon. The header `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` defines these functions within the namespace `mpp`. The `tensor_ops` namespace, which resides beneath the `mpp` namespace, contains functions that operate on tensors, including matrix multiplication and convolution. The functions that operate on tensors, Tensor Operations (TensorOps), use `tensor` and `cooperative_tensors` (see section 2.21) and have been tuned for Apple silicon GPUs. For a list of supported GPU families, refer to the Metal Feature Set Tables at developer.apple.com. When instantiating a TensorOp, you pass the scope of execution for the operation, where scope is the number of threads cooperating to execute the operation (see section 7.1).

## 7.1 Execution Scopes

All OS: Metal 4 and later support execution scopes.

Operations like TensorOps can work on a single thread, or cooperatively across threads in a SIMD-group or multiple SIMD-groups. You use execution scopes to specify the scope of cooperation. Table 7.1 outlines the types of execution scope.

<div align="center">

Table 7.1 Execution scopes

</div>

<table border="1"><tr><td>Scope</td><td>Description</td></tr><tr><td>execution_thread</td><td>Indicates the scope of cooperation is a single thread.</td></tr><tr><td>execution_simdgroups&lt;N&gt; or execution_simdgroup for N == 1</td><td>Indicates the scope of cooperation is N SIMD-groups. TensorOps support N with a value of 1 or `simdgroups_per_threadgroup` (see section 5.2.3.6). You can use `execution_simdgroup` for N = 1.</td></tr></table>

## 7.2 Tensor Operations (TensorOps)

All OS: Metal 4 and later support tensor operations (TensorOps).

TensorOps are GPU-accelerated functions that operate on tensors and cooperative_tensors (see section 2.21). TensorOps are class templates that you instantiate with a set of properties, including the execution scope, to indicate if the operation should run on a single thread or cooperatively across threads in a SIMD-group or multiple SIMD-groups (see section 7.1). When calling the TensorOp run method, all threads must call the method within that scope, or the result is undefined. For example, if the scope used to create the TensorOp is execution_simdgroup, you must ensure all threads within the same SIMD-group call the run method. Note that different SIMD-groups can be divergent with each other in this case.

TensorOps may use a barrier at the level of the execution scope. For example, if you specify the scope of an operation to be the entire threadgroup, you should ensure your code would behave correctly if a barrier is used in the TensorOp implementation.

If the TensorOps writes the result into a tensor whose ElementType is in device or threadgroup address space, you must insert a barrier (see section 6.9.1) at the appropriate thread scope and set the appropriate memory flags before reading the results. You don't need to use a barrier for tensors whose memory is in thread address space or for cooperative_tensors. For example, if the TensorOp run method writes to a tensor whose ElementType is in threadgroup memory and scope is execution_simdgroups<2>, call threadgroup_barrier(mem_flags::mem_threadgroup) before reading the result of the tensor. Another example is if the TensorOp run method writes to a tensor whose ElementType is in device memory and scope is execution_simdgroup, call simdgroup_barrier(mem_flags::mem_device) before reading the result of the tensor.

<div align="center">

Table 7.2 TensorOps

</div>

<table border="1"><tr><td>TensorOp template classes</td><td>Description</td></tr><tr><td>template&lt;matmul2d_descriptor Desc, typename Scope, class... Args&gt; matmul2d</td><td>Defines an object to perform a generalized matrix multiplication: C = A * B + C. A and B can be host-bound, origin-shifted, or shader-allocated tensors. C can be host-bound, origin-shifted, shader-allocated tensors, or `cooperative_tensor`. See section 7.2.1 for more details.</td></tr><tr><td>template&lt;convolution2d_descriptor Desc, typename Scope, typename... ConvArgs&gt; convolution2d</td><td>Defines an object to perform a 2D convolution that occurs in neural networks. 2D stands for two spatial dimensions of width x height. The tensor consumed by this op is 4D. The only scope currently supported is `execution_simdgroups&lt;N&gt;` where N is `simdgroups_per_threadgroup`. See section 7.2.2 for more details.</td></tr></table>

## 7.2.1 Matrix Multiplication

The template class matmul2d performs a generalized matrix multiplication of two tensors (C = A*B) or matrix multiplication accumulated into a tensor (C = A*B + C).

The operation takes an M x K tensor A multiplied by a K x N tensor B and accumulates it into an M x N tensor C. A and B can be host-bound, origin-shifted, or shader-allocated tensors. C can be host-bound, origin-shifted, shader-allocated tensors, or cooperative_tensor. Table 7.3 shows the data type combination supported.

<div align="center">

Table 7.3 MatMul2D data type supported

</div>

<table border="1"><tr><td>Tensor A type</td><td>Tensor B type</td><td>Tensor C type</td></tr><tr><td>char</td><td>char</td><td>int</td></tr><tr><td>char</td><td>half</td><td>half</td></tr><tr><td>char</td><td>half</td><td>float</td></tr><tr><td>char</td><td>float</td><td>float</td></tr><tr><td>half</td><td>char</td><td>half</td></tr><tr><td>half</td><td>char</td><td>float</td></tr><tr><td>half</td><td>half</td><td>half</td></tr><tr><td>half</td><td>half</td><td>float</td></tr><tr><td>half</td><td>float</td><td>float</td></tr><tr><td>float</td><td>char</td><td>float</td></tr><tr><td>float</td><td>half</td><td>float</td></tr><tr><td>float</td><td>float</td><td>float</td></tr></table>

<div align="center">

Table 7.4 shows additional data types supported in OS 26.1 and later.

</div>

<div align="center">

Table 7.4 Additional MatMul2D data types supported in OS 26.1 and later

</div>

<table border="1"><tr><td>Tensor A type</td><td>Tensor B type</td><td>Tensor C type</td></tr><tr><td>bfloat</td><td>bfloat</td><td>bfloat</td></tr><tr><td>bfloat</td><td>bfloat</td><td>float</td></tr><tr><td>bfloat</td><td>float</td><td>float</td></tr><tr><td>bfloat</td><td>char</td><td>bfloat</td></tr><tr><td>bfloat</td><td>char</td><td>float</td></tr><tr><td>float</td><td>bfloat</td><td>float</td></tr><tr><td>char</td><td>bfloat</td><td>bfloat</td></tr><tr><td>char</td><td>bfloat</td><td>float</td></tr><tr><td>bfloat</td><td>half</td><td>bfloat</td></tr><tr><td>bfloat</td><td>half</td><td>half</td></tr><tr><td>bfloat</td><td>half</td><td>float</td></tr><tr><td>half</td><td>bfloat</td><td>bfloat</td></tr><tr><td>half</td><td>bfloat</td><td>half</td></tr><tr><td>half</td><td>bfloat</td><td>float</td></tr></table>

To create the matmul2d, you first build a descriptor using the constructor below.

matmul2d_descriptor(int M, int N, int K = dynamic_length_v<int>,

bool transpose_left = false,

bool transpose_right = false,

bool relaxed_precision = false,

mode matmul_mode = mode::multiply);

<div align="center">

Table 7.5 MatMul2D descriptor parameters

</div>

<table border="1"><tr><td>Parameter</td><td>Description</td></tr><tr><td>M, N, K</td><td>Tensor dimensions where M x K tensor A, K x N tensor B, and M x N tensor C.</td></tr><tr><td>transpose_left</td><td>Transpose matrix A before multiplying. The default is false.</td></tr></table>

<table border="1"><tr><td>Parameter</td><td>Description</td></tr><tr><td>transpose_right</td><td>Transpose matrix B before multiplying. The default is false.</td></tr><tr><td>relaxed_precision</td><td>Specifies if the operation can use relaxed precision for float data type. Relaxed precision allows the operation to truncate the mantissa before the multiplication. The default is false.</td></tr><tr><td>matmul_mode</td><td>Specifies whether to perform a multiply or multiply_accumulate. The default is multiply.</td></tr></table>

<div align="center">

Table 7.6 MatMul2D member functions

</div>

<table border="1"><tr><td>MatMul2D member functions</td><td>Description</td></tr><tr><td>template&lt;typename LeftOperandType, typename RightOperandType, typename DestinationOperandType&gt; void run(thread LeftOperandType &amp;left, thread RightOperandType &amp;right, thread DestinationOperandType &amp;destination);</td><td>Executes a matrix multiply of C = A * B where C is the destination tensor, A is the left tensor, and B is the right tensor.</td></tr><tr><td>template&lt;typename LeftOperandType, typename RightOperandType, typename ElementType, typename CoordType = int&gt; cooperative_tensor&lt;...&gt; get_destination_cooperative_tensor() thread const;</td><td>Returns a cooperative_tensor that can store the result of the matrix multiply.</td></tr><tr><td>template&lt;typename LeftOperandType, typename RightOperandType, typename ElementType, typename CoordType = int&gt; cooperative_tensor&lt;...&gt; get_row_reduction_destination_cooperative_tensor() thread const;</td><td>Returns a cooperative_tensor that can store the result of the row reduction on the result of the matrix multiply.</td></tr><tr><td>template&lt;typename LeftOperandType, typename RightOperandType, typename ElementType, typename CoordType = int&gt; cooperative_tensor&lt;...&gt; get_column_reduction_destination_cooperative_tensor() thread const;</td><td>Returns a cooperative_tensor that can store the result of the column reduction on the result of the matrix multiply.</td></tr></table>

To instantiate the template matmul2d, you pass the descriptor and the execution scope to the template:

```xml

template < matmul2d_descriptor Desc,

    typename Scope,

    class... Args>  matmul2d;

```

To execute the matrix multiplication, call the matmul2d run method by passing the left tensor (A), the right tensor (B), and the destination tensor (C):

```cpp

template <

    typename LeftOperandType,

    typename RightOperandType,

    typename DestinationOperandType>

void run(thread LeftOperandType &left,

        thread RightOperandType &right,

        thread DestinationOperandType &destination);

```

See Table 7.3 and Table 7.4 for the element type supported for tensor A, B, C.

The example below illustrates the use of a matmul2d TensorOp with tensors:

```cpp

#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

using namespace mpp;

[[ kernel ]] void matrixMultiply(

  tensor<device half, dextents<int, 2>> a [[ buffer(0) ]],

  tensor<device half, dextents<int, 2>> b [[ buffer(1) ]],

  tensor<device half, dextents<int, 2>> c [[ buffer(2) ]],

  uint2 tgid [[thread_position_in_grid]]) {

  // Create a matmul op for a threadgroup made of 4 SIMD-groups.

  constexpr auto matmulDescriptor =

    tensor_ops::matmul2d_descriptor(64, 32, 0);

  tensor_ops::matmul2d<matmulDescriptor,

                    execution_simdgroups<4>> matmulOp;

  // Create the appropriate slice for this threadgroup to work on.

  auto mA = a.slice(0, tgid.y * 64);

  auto mB = b.slice(tgid.x * 32, 0);

  auto mC = c.slice(tgid.x * 32, tgid.y * 64);

  // Execute the operation assuming C is initialized to zero.

  matmulOp.run(mA, mB, mC);

}

To use a cooperative_tensor for the destination of a matmul2d TensorOp, use the following member function. The function returns a cooperative_tensor whose storage is divided across the threads in the scope of the matmul2d:

```cpp

template <typename LeftOperandType,

          typename RightOperandType,

          typename ElementType, typename CoordType = int>

cooperative_tensor<...>

get_destination_cooperative_tensor() thread const;

```

The example below illustrates the use of a matmul2d TensorOp with cooperative_tensor:

```cpp

#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

using namespace mpp;


[[ kernel ]] void gemmBias(

  tensor<device float, dextents<int, 2>> a [[ buffer(0) ]],

  tensor<device float, dextents<int, 2>> b [[ buffer(1) ]],

  tensor<device float, dextents<int, 2>> c [[ buffer(2) ]],

  device float*                        bufBias [[buffer(3)]],

  uint2 tgid [[thread_position_in_grid]]) {

  // Build the bias tensor from the buffer.

  array<int,1> stride = {1};

  tensor<device float, dextents<int, 1>, tensor_inline>

    tBias(bufBias, dextents<int,1>(64), stride);

  // Create a matmul op for a threadgroup made of 4 SIMD-groups.

  constexpr auto matmulDescriptor =

    tensor_ops::matmul2d_descriptor(

      64, 32, 0, false, false, false,

      tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  tensor_ops::matmul2d<matmulDescriptor,

                    execution_simdgroups<4>> matmulOp;

  // Create the cooperative tensor.

  auto cTc = matmulOp.get_destination_cooperative_tensor<

                decltype(a), decltype(b), float>();

  // Load the bias, run the matrix multiple and store the result.

  cTc.load(tBias);

  matmulOp.run(a, b, cTc);

  cTc.store(c);

}

```

You can do a row or column sum, max, or min reduction of a cooperative_tensor into a destination 1D cooperative_tensor if the scope of the matmul2d is execution_simdgroup.

<div align="center">

Table 7.7 Reduction related functions for cooperative tensors

</div>

<table border="1"><tr><td>Reduction related functions</td><td>Description</td></tr><tr><td>template&lt;class ElementType,class SrcExtents,class DstExtents,class SrcLayout,class DstLayout&gt;inline void reduce_rows(thread metal::cooperative_tensor&lt;ElementType, SrcExtents,SrcLayout&gt; &amp;source,thread metal::cooperative_tensor&lt;ElementType, DstExtents,DstLayout&gt; &amp;destination,reduction_operation op=reduction_operation::sum,ElementType identity=reduction_operation_identity&lt;ElementType&gt;::sum_identity);</td><td>Returns the reduction of each row and stores the result into the 1D destinationcooperative_tensor.The default is a sum reduction for each row.</td></tr><tr><td>template&lt;class ElementType,class SrcExtents,class DstExtents,class SrcLayout,class DstLayout&gt;inline void reduce_columns(thread metal::cooperative_tensor&lt;ElementType, SrcExtents,SrcLayout&gt; &amp;source,thread metal::cooperative_tensor&lt;ElementType, DstExtents,DstLayout&gt; &amp;destination,reduction_operation op=reduction_operation::sum,ElementType identity=reduction_operation_identity&lt;ElementType&gt;::sum_identity);</td><td>Returns the reduction of each column and stores the result into the 1D destinationcooperative_tensor.The default is a sum reduction for each column.</td></tr></table>

```cpp

template <

    class SrcElementType,

    class DstElementType,

    class SrcExtents,

    class DstExtents,

    class SrcLayout,

    class DstLayout>

inline bool is_iterator_compatible(

    const thread metal::cooperative_tensor<

        SrcElementType,

        SrcExtents,

        SrcLayout> &source,

    const thread metal::cooperative_tensor<

        DstElementType,

        DstExtents,

        DstLayout> &destination);

```

Returns true if you can use the result of the reduction with another tensor using the map_iterator. To check if the iterators are compatible, call the following nonmember function.

To get the destination tensor for a row reduction, call the following member function:

```cpp

template <typename LeftOperandType,

    typename RightOperandType,

    typename ElementType, typename CoordType = int>

cooperative_tensor<...>

get_row_reduction_destination_cooperative_tensor() thread const;

```

To get the destination tensor for a column reduction, call the following member function:

```cpp

template <typename LeftOperandType,

    typename RightOperandType,

    typename ElementType, typename CoordType = int>

cooperative_tensor<...>

get_column_reduction_destination_cooperative_tensor() thread

const;

```

Use the enumeration to define the type of reduction:

```cpp

enum class reduction_operation {

    sum, // Take the sum of the element of the row/column.

    max, // Take the max value of all elements in row/column.

    min, // Take the min value of all elements in row/column.

};

```

Use the following structure to define the identity value for the type of reduction:

```cpp

template <typename ElementType>

struct reduction_operation_identity

{

    static const constant ElementType sum_identity;

    static const constant ElementType max_identity;

    static const constant ElementType min_identity;

};

```

Call the following nonmember function to return the reduction of each row and store the result into the 1D destination cooperative_tensor. The default is a sum reduction for each row.

```cpp

template <class ElementType, class SrcExtents,

    class DstExtents, class SrcLayout,

    class    DstLayout>

inline void reduce_rows(

  thread metal::cooperative_tensor<ElementType, SrcExtents,

    SrcLayout> &source,

  thread metal::cooperative_tensor<ElementType, DstExtents,

    DstLayout> &destination,

  reduction_operation op = reduction_operation::sum,

  ElementType identity =

    reduction_operation_identity<ElementType>::sum_identity);

```

Call the following nonmember function to return the reduction of each column and store the result into the 1D destination cooperative_tensor. The default is a sum reduction for each column.

```cpp

template <class ElementType, class SrcExtents, class DstExtents,

    class SrcLayout, class DstLayout>

inline void reduce_columns(

    thread metal::cooperative_tensor<ElementType, SrcExtents,

        SrcLayout> &source,

    thread metal::cooperative_tensor<ElementType, DstExtents,

        DstLayout> &destination,

    reduction_operation op = reduction_operation::sum,

    ElementType identity =

        reduction_operation_identity<ElementType>::sum_identity);

```

The example below demonstrates how to do a row reduction:

```cpp

[[ kernel ]] void gemm_reduce(

tensor<device float, dextents<int, 2>> aT [[ buffer(0) ]],

tensor<device float, dextents<int, 2>> bT [[ buffer(1) ]],

tensor<device float, dextents<int, 2>> cT [[ buffer(2) ]],

tensor<device float, dextents<int, 1>> dR [[ buffer(3) ]],

uint2 tgid [[thread_position_in_grid]]) {

constexpr auto matmulDescriptor =

    tensor_ops::matmul2d_descriptor(64, 32, 0);

tensor_ops::matmul2d<matmulDescriptor,

    execution_simdgroup> matmulOp;

// Create the cooperative tensor.

auto cTdest = matmulOp.get_destination_cooperative_tensor<

    decltype(aT), decltype(bT), float>();

// Run the matrix multiple.

matmulOp.run(aT, bT, cTdest);

// Sum up each row and store the results.

auto cTred =

    matmulOp.get_row_reduction_destination_cooperative_tensor<

    decltype(aT), decltype(bT), float>();

reduce_rows(cTdest, cTred, tensor_ops::reduction_operation::sum,

    0.0f);

cTred.store(dR);

}

```

You can use the result of the reduction with another tensor using the map_iterator. To check if the iterators are compatible, call the following nonmember function.

```cpp

template <class SrcElementType, class DstElementType,

    class SrcExtents, class DstExtents,

    class SrcLayout, class DstLayout>

inline bool is_iterator_compatible(

    const thread metal::cooperative_tensor<

        SrcElementType,

        SrcExtents,

        SrcLayout> &source,

    const thread metal::cooperative_tensor<

        DstElementType,

        DstExtents,

        DstLayout> &destination);

```

The following example shows a use of is_iterator_compatible and map_iterator:

```cpp

[[ kernel ]] void gemm_map(

  tensor<device float, dextents<int, 2>> aT [[ buffer(0) ]],

  tensor<device float, dextents<int, 2>> bT [[ buffer(1) ]],

  tensor<device float, dextents<int, 2>> dT [[ buffer(2) ]])

{

  constexpr auto matmulDescriptor =

    tensor_ops::matmul2d_descriptor(64, 32, 0);

  tensor_ops::matmul2d<matmulDescriptor,

                    execution_simdgroup> matmulOp;

  // Create the cooperative tensor.

  auto cTdest = matmulOp.get_destination_cooperative_tensor<

                decltype(aT), decltype(bT), float>();

  // Load the bias, run the matrix multiple, and store the result.

  matmulOp.run(aT, bT, cTdest);

  auto cTred =

    matmulOp.get_row_reduction_destination_cooperative_tensor<

                decltype(aT), decltype(bT), float>();

  auto identity = metal::numeric_limits<float>::lowest();

  reduce_rows(cTdest, cTred, tensor_ops::reduction_operation::min,

              identity);

  // Check if the iterators are compatible and if so, add

  // the min across the rows.

  if (tensor_ops::is_iterator_compatible(cTdest, cTred)) {

    for (auto it = cTdest.begin(); it != cTdest.end(); it++) {

      auto cTred_it = cTred.map_iterator(it);

      *it += *cTred_it;

    }

  }

  else {

    // Do something else.

  }

  cTdest.store(dT);

}

```

For more detailed information, see the `MPPTensorOpsMatMul2d.h` header.

## 7.2.2 Convolution

The template class convolution2d performs a 2D convolution where 2D stands for two spatial dimensions of width x height. The operation takes an activation and a weight tensor to produce a tensor or cooperative_tensor as described in Table 7.8.

To create a convolution2d, you first build a descriptor using the constructor below:

```cpp

enum class convolution2d_activation_layout {

    nhwc,

};

enum class convolution2d_weights_layout {

    hwio,

};

convolution2d_descriptor(

    int4 destination_dimensions,

    int4 source_dimensions,

    int2 kernel_dimensions,

    convolution2d_activation_layout activation_layout =

        convolution2d_activation_layout::nhwc,

    convolution2d_weights_layout weight_layout =

        convolution2d_weights_layout::hwio,

    int2 strides = int2(1, 1),

    int2 dilations = int2(1, 1),

    int  groups = 1,

    bool relaxed_precision = false,

    mode convolution2d_mode = mode::multiply);

```

<div align="center">

Table 7.8 Convolution2d parameters

</div>

<table border="1"><tr><td>Parameter</td><td>Description</td></tr><tr><td>destination_dimensions</td><td>Specifies the dimension of the output tensor.</td></tr><tr><td>source_dimensions</td><td>Specifies the dimension of the input tensor.</td></tr><tr><td>kernel_dimensions</td><td>Specifies the size of the convolution window.</td></tr><tr><td>activation_layout</td><td>Specifies the layout of the activation tensor.</td></tr><tr><td>weights_layout</td><td>Specifies the layout of the weight tensor.</td></tr><tr><td>strides</td><td>Specifies the stride of the convolution</td></tr><tr><td>dilations</td><td>Specifies the spacing between kernel elements.</td></tr></table>

<table border="1"><tr><td>Parameter</td><td>Description</td></tr><tr><td>groups</td><td>Specifies the number of groups the input is split to the channel axis.</td></tr><tr><td>relaxed_precision</td><td>Specifies if the operation can use relaxed precision for float data type. Relaxed precision allows the operation to truncate the mantissa before the multiplication.</td></tr><tr><td>convolution2d_mode</td><td>Specifies whether to perform a multiply or multiply_accumulate.</td></tr></table>

To instantiate the template convolution2d, you pass the descriptor and scope. Currently, the only scope supported is execution_simdgroups<N> where N is simdgroups_per_threadgroup.

```html

template <

  convolution2d_descriptor Desc,

  typename Scope,

  typename... ConvArgs>

convolution2d;

```

To execute the convolution, call the convolution2d run method:

```cpp

template <typename ActivationTensorType,

    typename WeightsTensorType,

    typename DestinationTensorType, typename... RunArgs>

void run(thread ActivationTensorType &activation,

    thread WeightsTensorType &weights,

    thread DestinationTensorType &destination) const;

```

<div align="center">

Table 7.9 Convolution run parameter

</div>

<table border="1"><tr><td>Parameter</td><td>Description</td></tr><tr><td>activation</td><td>The activation tensor with NHWC layout:
N=batch(slowest moving dimension)
H=height
W=width
C=input channels(fastest moving dimension)</td></tr><tr><td>weights</td><td>The weights tensor with HWIO layout:
H=kernel height
W=kernel width
I=input channels
O=output channels(fastest moving dimension)</td></tr></table>

<table border="1"><tr><td>Parameter</td><td>Description</td></tr><tr><td>destination</td><td>The destination tensor which can be a tensor or a cooperative tensor.If it is a tensor,the format is NHWO layout:N=batch(slowest moving dimension)H=heightW=widthO=output channels(fastest moving dimension)</td></tr></table>

For more detailed information, please see the MPPTensorOpsConvolution2d.h header.
