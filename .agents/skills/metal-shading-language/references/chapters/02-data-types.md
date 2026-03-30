## 2 Data Types

This chapter details the Metal data types, including types that represent vectors and matrices. The chapter also discusses atomic data types, buffers, textures, samplers, arrays, user-defined structures, type alignment, and type conversion.

## 2.1 Scalar Data Types

Metal supports the scalar types listed in Table 2.1. Metal does not support the double, long long, unsigned long long, and long double data types.

<div align="center">

Table 2.1. Metal scalar data types

</div>

<table border="1"><tr><td>Type</td><td>Description</td></tr><tr><td>bool</td><td>A conditional data type that has the value of either true or false. The value true expands to the integer constant 1, and the value false expands to the integer constant 0.</td></tr><tr><td>char int8_t</td><td>A signed two&#x27;s complement 8-bit integer.</td></tr><tr><td>unsigned char uchar uint8_t</td><td>An unsigned 8-bit integer.</td></tr><tr><td>short int16_t</td><td>A signed two&#x27;s complement 16-bit integer.</td></tr><tr><td>unsigned short ushort uint16_t</td><td>An unsigned 16-bit integer.</td></tr><tr><td>int int32_t</td><td>A signed two&#x27;s complement 32-bit integer.</td></tr><tr><td>unsigned int uint uint32_t</td><td>An unsigned 32-bit integer.</td></tr><tr><td>long int64_t</td><td>A signed two's complement 64-bit integer. All OS: Metal 2.2 and later</td></tr><tr><td>unsigned long uint64_t</td><td>An unsigned 64-bit integer. All OS: Metal 2.2 and later</td></tr><tr><td>half</td><td>A 16-bit floating-point. The half data type must conform to the IEEE 754 binary16 storage format.</td></tr><tr><td>bfloat</td><td>A 16-bit bfloat floating-point. The bfloat data type is a truncated version of float for machine learning applications, using an 8-bit (7 explicitly stored) rather than 24-bit mantissa. All OS: Metal 3.1 and later</td></tr><tr><td>float</td><td>A 32-bit floating-point. The float data type must conform to the IEEE 754 single precision storage format.</td></tr><tr><td>size_t</td><td>An unsigned integer type of the result of the sizeof operator. This is a 64-bit unsigned integer.</td></tr><tr><td>ptrdiff_t</td><td>A signed integer type that is the result of subtracting two pointers. This is a 64-bit signed integer.</td></tr><tr><td>void</td><td>The void type comprises an empty set of values; it is an incomplete type that cannot be completed.</td></tr></table>

## Metal supports:

- the f or F suffix to specify a single precision floating-point literal value (such as 0.5f or 0.5F).

- the h or H suffix to specify a half precision floating-point literal value (such as 0.5h or 0.5H).

- the bf or BF suffix to specify a bfloat precision floating-point literal value (such as 0.5bf or 0.5BF).

- the u or U suffix for unsigned integer literals.

- the l or L suffix for signed long integer literals.

<div align="center">

Table 2.2 lists the size and alignment of most of the scalar data types.

</div>

<div align="center">

Table 2.2. Size and alignment of scalar data types

</div>

<table border="1"><tr><td>Type</td><td>Size(in bytes)</td><td>Alignment(in bytes)</td></tr><tr><td>bool</td><td>1</td><td>1</td></tr></table>

<table border="1"><tr><td>Type</td><td>Size(in bytes)</td><td>Alignment(in bytes)</td></tr><tr><td>char
int8_t
unsigned char
uchar
uint8_t</td><td>1</td><td>1</td></tr><tr><td>short
int16_t
unsigned short
ushort
uint16_t</td><td>2</td><td>2</td></tr><tr><td>int
int32_t
unsigned int
uint
uint32_t</td><td>4</td><td>4</td></tr><tr><td>long
int64_t
unsigned long
uint64_t</td><td>8</td><td>8</td></tr><tr><td>size_t</td><td>8</td><td>8</td></tr><tr><td>half</td><td>2</td><td>2</td></tr><tr><td>bfloat</td><td>2</td><td>2</td></tr><tr><td>float</td><td>4</td><td>4</td></tr></table>

## 2.2 Vector Data Types

Metal supports a subset of the vector data types implemented by the system vector math library. Metal supports these vector type names, where n is 2, 3, or 4, representing a 2-, 3-, or 4-component vector type, respectively:

- booln

- charn

- shortn

- intn

- longn

- ucharn

- ushortn

- uintn

- ulongn

- halfn

- bfloatn (Metal 3.1 and later)

- floatn

Metal also supports vec<T, n> where T is a valid scalar type and n is 2, 3, or 4, representing a 2-, 3-, or 4-component vector type.

Table 2.3 lists the size and alignment of the vector data types.

<div align="center">

Table 2.3. Size and alignment of vector data types

</div>

<table border="1"><tr><td>Type</td><td>Size(in bytes)</td><td>Alignment(in bytes)</td></tr><tr><td>bool2</td><td>2</td><td>2</td></tr><tr><td>bool3</td><td>4</td><td>4</td></tr><tr><td>bool4</td><td>4</td><td>4</td></tr><tr><td>char2
uchar2</td><td>2</td><td>2</td></tr><tr><td>char3
uchar3</td><td>4</td><td>4</td></tr><tr><td>char4
uchar4</td><td>4</td><td>4</td></tr><tr><td>short2
ushort2</td><td>4</td><td>4</td></tr><tr><td>short3
ushort3</td><td>8</td><td>8</td></tr><tr><td>short4
ushort4</td><td>8</td><td>8</td></tr><tr><td>int2
uint2</td><td>8</td><td>8</td></tr><tr><td>int3
uint3</td><td>16</td><td>16</td></tr><tr><td>int4
uint4</td><td>16</td><td>16</td></tr><tr><td>long2
ulong2</td><td>16</td><td>16</td></tr></table>

<table border="1"><tr><td>Type</td><td>Size(in bytes)</td><td>Alignment(in bytes)</td></tr><tr><td>long3
ulong3</td><td>32</td><td>32</td></tr><tr><td>long4
ulong4</td><td>32</td><td>32</td></tr><tr><td>half2</td><td>4</td><td>4</td></tr><tr><td>half3</td><td>8</td><td>8</td></tr><tr><td>half4</td><td>8</td><td>8</td></tr><tr><td>bfloat2</td><td>4</td><td>4</td></tr><tr><td>bfloat3</td><td>8</td><td>8</td></tr><tr><td>bfloat4</td><td>8</td><td>8</td></tr><tr><td>float2</td><td>8</td><td>8</td></tr><tr><td>float3</td><td>16</td><td>16</td></tr><tr><td>float4</td><td>16</td><td>16</td></tr></table>

## 2.2.1 Accessing Vector Components

You can use an array index to access vector components. Array index 0 refers to the first component of the vector, index 1 to the second component, and so on. The following examples show various ways to access array components:

```c

pos = float4(1.0f, 2.0f, 3.0f, 4.0f);

float x = pos[0]; // x = 1.0

float z = pos[2]; // z = 3.0

float4 vA = float4(1.0f, 2.0f, 3.0f, 4.0f);

float4 vB;

for (int i=0; i<4; i++)

    vB[i] = vA[i] * 2.0f; // vB = (2.0, 4.0, 6.0, 8.0);
}

```

Metal supports using a period (. ) as a selection operator to access vector components, using letters that may indicate coordinate or color data:

```cpp

<vector_data_type>.xyzw

<vector_data_type>.rgba

```

The following code initializes a vector test and then uses the .xyzw or .rgba selection syntax to access individual components:

```cpp

int4 test = int4(0, 1, 2, 3);

int a = test.x; // a = 0

int b = test.y; // b = 1

int c = test.z; // c = 2

int d = test.w; // d = 3

int e = test.r; // e = 0

int f = test.g; // f = 1

int g = test.b; // g = 2

int h = test.a; // h = 3

```

The component selection syntax allows the selection of multiple components:

```cpp
float4 c;

c.xyzw = float4(1.0f, 2.0f, 3.0f, 4.0f);

c.z = 1.0f;

c.xy = float2(3.0f, 4.0f);

c.xyz = float3(3.0f, 4.0f, 5.0f);
```

The component selection syntax also allows the permutation or replication of components:

```cpp
float4 pos = float4(1.0f, 2.0f, 3.0f, 4.0f);

float4 swiz = pos.wzyx; // swiz = (4.0f, 3.0f, 2.0f, 1.0f)

float4 dup = pos.xxyy; // dup = (1.0f, 1.0f, 2.0f, 2.0f)
```

The component group notation can occur on the left-hand side (lvalue) of an expression. To form the lvalue, you may apply swizzling. The resulting lvalue may be either the scalar or vector type, depending on number of components specified. Each component must be a supported scalar or vector type. The resulting lvalue of vector type must not contain duplicate components.

```cpp
float4 pos = float4(1.0f, 2.0f, 3.0f, 4.0f);

// pos = (5.0, 2.0, 3.0, 6.0)
pos.xw = float2(5.0f, 6.0f);

// pos = (8.0, 2.0, 3.0, 7.0)
pos.wx = float2(7.0f, 8.0f);

// pos = (3.0, 5.0, 9.0, 7.0)
pos.xyz = float3(3.0f, 5.0f, 9.0f);
```

When assigning a swizzled value to a variable, the GPU may need to read the existing value, modify it, and write the result back. The assignment to pos.xw in the example above causes the GPU to load the float4 value, shuffle values 5.0f and 6.0f into it, and then write the result back into pos. If two threads write to different components of the vector at the same time, the result is undefined.

The following methods of vector component access are not permitted and result in a compile-time error:

- Accessing components beyond those declared for the vector type is an error. 2-component vector data types can only access .xy or .rg elements. 3-component vector data types can only access .xyz or .rgb elements.

float2 pos; // This is a 2-component vector.

pos.x = 1.0f; // x is legal and so is y.

pos.z = 1.0f; // z is illegal and so is w. z is the 3rd component.

float3 pos; // This is a 3-component vector.

pos.z = 1.0f; // z is legal for a 3-component vector.

pos.w = 1.0f; // This is illegal. w is the 4th component.

- Accessing the same component twice on the left-hand side is ambiguous and is an error:

  // This is illegal because 'x' is used twice.

  pos.xx = float2(3.0f, 4.0f);

- Accessing a different number of components is an error:

  // This is illegal due to a mismatch between float2 and float4.

  pos.xy = float4(1.0f, 2.0f, 3.0f, 4.0f);

- Intermixing the .rgba and .xyzw syntax in a single access is an error:

  float4 pos = float4(1.0f, 2.0f, 3.0f, 4.0f);

  pos.x = 1.0f; // OK

  pos.g = 2.0f; // OK

  // These are illegal due to mixing rgba and xyzw attributes.

```cpp
pos.xg = float2(3.0f, 4.0f);

float3 coord = pos.ryz;
```

- A pointer or reference to a vector with swizzles is an error:

```cpp
float4 pos = float4(1.0f, 2.0f, 3.0f, 4.0f);

my_func(&pos.xy); // This is an illegal pointer to a swizzle.
```

The sizeof operator on a vector type returns the size of the vector. This is typically the number of components * size of each component, except for 3-component vectors whose size is the same as the 4-component vector (see Table 2.3). For example, sizeof(float4) returns 16 and sizeof(half4) returns 8.

## 2.2.2 Vector Constructors

You can use constructors to create vectors from a set of scalars or vectors. The parameter signature determines how to construct and initialize a vector. For instance, if the vector is initialized with only a single scalar parameter, all components of the constructed vector are set to that scalar value.

If you construct a vector from multiple scalars, one or more vectors, or a mixture of scalars and vectors, Metal consumes the vector's components in order from the components of the arguments. Metal consumes the arguments from left to right. Metal consumes all of an argument's components, in order, before any components from the following argument.

float4(float x);

float4(float x, float y, float z, float w);

float4(float2 a, float2 b);

float4(float2 a, float b, float c);

float4(float a, float b, float2 c);

float4(float a, float2 b, float c);

float4(float3 a, float b);

float4(float a, float3 b);

float4(float4 x);

This is a list of constructors for float3:

float3(float x);

float3(float x, float y, float z);

float3(float a, float2 b);

float3(float2 a, float b);

float3(float3 x);

This is a list of constructors for float2:

float2(float x);

float2(float x, float y);

float2(float2 x);

The following examples illustrate uses of the constructors:

float x = 1.0f, y = 2.0f, z = 3.0f, w = 4.0f;

float4 a = float4(0.0f);

float4 b = float4(x, y, z, w);

float2 c = float2(5.0f, 6.0f);

float2 a = float2(x, y);

float2 b = float2(z, w);

float4 x = float4(a.xy, b.xy);

Under-initializing a vector constructor results in a compile-time error.

## 2.2.3 Packed Vector Types

You must align the vector data types described in section 2.2 to the size of the vector. You can also require their vector data to be tightly packed; for example, a vertex structure that may contain position, normal, tangent vectors and texture coordinates tightly packed and passed as a buffer to a vertex function.

The supported packed vector type names are:

- packed_charn

- packed_shortn

- packed_intn

- packed_ucharn

- packed_ushortn

- packed_uintn

- packed_halfn

- packed_bfloatn (Metal 3.1 and later)

- packed_floatn

- packed_longn (Metal 2.3 and later)

Where n is 2,3,or 4,representing a 2-,3-,or 4-component vector type,respectively. (The packed_booln vector type names are reserved.)

Metal also supports packed_vec<T,n> where T is a valid scalar type and n is 2,3,or 4, representing a 2-, 3-, or 4-component packed vector type.

Table 2.4 lists the size and alignment of the packed vector data types.

<div align="center">

Table 2.4. Size and alignment of packed vector data types

</div>

<table border="1"><tr><td>Type</td><td>Size(in bytes)</td><td>Alignment(in bytes)</td></tr><tr><td>packed_char2,packed_uchar2</td><td>2</td><td>1</td></tr><tr><td>packed_char3,packed_uchar3</td><td>3</td><td>1</td></tr><tr><td>packed_char4,packed_uchar4</td><td>4</td><td>1</td></tr><tr><td>packed_short2,packed_ushort2</td><td>4</td><td>2</td></tr><tr><td>packed_short3,packed_ushort3</td><td>6</td><td>2</td></tr><tr><td>packed_short4,packed_ushort4</td><td>8</td><td>2</td></tr><tr><td>packed_int2,packed_uint2</td><td>8</td><td>4</td></tr><tr><td>packed_int3,packed_uint3</td><td>12</td><td>4</td></tr><tr><td>packed_int4,packed_uint4</td><td>16</td><td>4</td></tr><tr><td>packed_half2</td><td>4</td><td>2</td></tr><tr><td>packed_half3</td><td>6</td><td>2</td></tr><tr><td>packed_half4</td><td>8</td><td>2</td></tr><tr><td>packed_bfloat2</td><td>4</td><td>2</td></tr><tr><td>packed_bfloat3</td><td>6</td><td>2</td></tr><tr><td>packed_bfloat4</td><td>8</td><td>2</td></tr><tr><td>packed_float2</td><td>8</td><td>4</td></tr><tr><td>packed_float3</td><td>12</td><td>4</td></tr></table>

<table border="1"><tr><td>Type</td><td>Size(in bytes)</td><td>Alignment(in bytes)</td></tr><tr><td>packed_float4</td><td>16</td><td>4</td></tr><tr><td>packed_long2</td><td>16</td><td>8</td></tr><tr><td>packed_long3</td><td>24</td><td>8</td></tr><tr><td>packed_long4</td><td>32</td><td>8</td></tr></table>

Packed vector data types are typically used as a data storage format. Metal supports the assignment, arithmetic, logical, relational, and copy constructor operators for packed vector data types. Metal also supports loads and stores from a packed vector data type to an aligned vector data type and vice-versa.

Examples:

```c

device float4 *buffer;

device packed_float4 *packed_buffer;

int i;

packed_float4 f ( buffer[i] );

packed_buffer[i] = buffer[i];

// An operator used to convert from packed_float4 to float4.

buffer[i] = float4( packed_buffer[i] );

```

You can use an array index to access components of a packed vector data type. In Metal 2.1 and later, you can use .xyzw or .rgba selection syntax to access components of a packed vector data type. The semantics and restrictions when swizzling for packed vector data type are the same as for vector types.

Example:

packed_float4 f;

f[0] = 1.0f; // OK

f.x = 1.0f; // OK, Metal 2.1 and later.

## 2.3 Matrix Data Types

Metal supports a subset of the matrix data types implemented by the system math library.

The supported matrix type names are:

- halfnxm

- floatnxm

Where n and m are numbers of columns and rows. n and m must be 2,3,or 4. A matrix of type floatnxm consists of n floatm vectors. Similarly, a matrix of type halfnxm consists of n halfm vectors.

Metal also supports matrix<T,c,r>,where T is a valid floating-point type, c is 2,3,or 4,and r is 2,3,or 4.

Table 2.5 lists the size and alignment of the matrix data types.

<div align="center">

Table 2.5. Size and alignment of matrix data types

</div>

<table border="1"><tr><td>Type</td><td>Size(in bytes)</td><td>Alignment(in bytes)</td></tr><tr><td>half2x2</td><td>8</td><td>4</td></tr><tr><td>half2x3</td><td>16</td><td>8</td></tr><tr><td>half2x4</td><td>16</td><td>8</td></tr><tr><td>half3x2</td><td>12</td><td>4</td></tr><tr><td>half3x3</td><td>24</td><td>8</td></tr><tr><td>half3x4</td><td>24</td><td>8</td></tr><tr><td>half4x2</td><td>16</td><td>4</td></tr><tr><td>half4x3</td><td>32</td><td>8</td></tr><tr><td>half4x4</td><td>32</td><td>8</td></tr><tr><td>float2x2</td><td>16</td><td>8</td></tr><tr><td>float2x3</td><td>32</td><td>16</td></tr><tr><td>float2x4</td><td>32</td><td>16</td></tr><tr><td>float3x2</td><td>24</td><td>8</td></tr><tr><td>float3x3</td><td>48</td><td>16</td></tr><tr><td>float3x4</td><td>48</td><td>16</td></tr><tr><td>float4x2</td><td>32</td><td>8</td></tr><tr><td>float4x3</td><td>64</td><td>16</td></tr><tr><td>float4x4</td><td>64</td><td>16</td></tr></table>

## 2.3.1 Accessing Matrix Components

You can use the array subscripting syntax to access the components of a matrix. Applying a single subscript to a matrix treats the matrix as an array of column vectors. Two subscripts select a column and then a row. The top column is column 0. A second subscript then operates on the resulting vector, as defined earlier for vectors.

float4x4 m;

// This sets the 2nd column to all 2.0.

m[1] = float4(2.0f);

// This sets the 1st element of the 1st column to 1.0.

m[0][0] = 1.0f;

// This sets the 4th element of the 3rd column to 3.0.

m[2][3] = 3.0f;

Access floatnxm and halfnxm matrices as an array of n floatm or n halfm entries.

Accessing a component outside the bounds of a matrix with a nonconstant expression results in undefined behavior. Accessing a matrix component that is outside the bounds of the matrix with a constant expression generates a compile-time error.

## 2.3.2 Matrix Constructors

Use constructors to create matrices from a set of scalars, vectors, or matrices. The parameter signature determines how to construct and initialize a matrix. For example, if you initialize a matrix with only a single scalar parameter, the result is a matrix that contains that scalar for all components of the matrix's diagonal, with the remaining components initialized to 0.0. For example, a call to:

float4x4(fval);

Where fval is a scalar floating-point value constructs a matrix with these initial contents:

fval 0.0 0.0 0.0

0.0 fval 0.0 0.0

0.0 0.0 fval 0.0

0.0 0.0 0.0 fval

You can also construct a matrix from another matrix that has the same number of rows and columns. For example:

float3x4(float3x4);

float3x4(half3x4);

Metal constructs and consumes matrix components in column-major order. The matrix constructor needs to have just enough specified values in its arguments to initialize every component in the constructed matrix object. Providing more arguments than necessary results in an error. Under-initializing a matrix constructor results in a compile-time error.

You can also construct a matrix of type T with n columns and m rows from n vectors of type T with m components. The following examples are legal constructors:

float2x2(float2, float2);

float3x3(float3, float3, float3);

float3x2(float2, float2, float2);

In Metal 2 and later, a matrix of type T with n columns and m rows can also be constructed from n * m scalars of type T. The following examples are legal constructors:

float2x2(float, float, float, float);

float3x2(float, float, float, float, float, float);

The following are examples of matrix constructors that Metal doesn't support. You can't construct a matrix from combinations of vectors and scalars.

// Not supported.

float2x3(float2 a, float b, float2 c, float d);

## 2.4 SIMD-group Matrix Data Types

All OS: Metal 2.3 and later support SIMD-group matrix types.

Metal supports a matrix type simdgroup_matrix<T,Cols,Rows> defined in <metal_simdgroup_matrix>. Operations on SIMD-group matrices are executed cooperatively by threads in the SIMD-group. Therefore, all operations must be executed only under uniform control-flow within the SIMD-group or the behavior is undefined.

Metal supports the following SIMD-group matrix type names, where T is half, bfloat (in Metal 3.1 and later) or float and Cols and Rows are 8:

- simdgroup_half8x8

- simdgroup_bfloat8x8 (Metal 3.1 and later)

- simdgroup_float8x8

The mapping of matrix elements to threads in the SIMD-group is unspecified. For a description of which functions Metal supports on SIMD-group matrices, see section 6.7

## 2.5 Alignment of Data Types

You can use the alignas alignment specifier to specify the alignment requirement of a type or an object. You may also apply the alignas specifier to the declaration of a variable or a data member of a structure or class. You may also apply it to the declaration of a structure, class, or enumeration type.

The Metal compiler is responsible for aligning data items to the appropriate alignment as required by the data type. For arguments to a graphics or kernel function declared to be a pointer to a data type, the Metal compiler assumes that the object referenced by the pointer is always appropriately aligned as required by the data type.

## 2.6 Atomic Data Types

Objects of atomic types are free from data races. If one thread writes to an atomic object while another thread reads from it, the behavior is well-defined.

Metal supports atomic<T>, where T can be int, uint, bool, or ulong for all OSes that support Metal 2.4 and later, or T can be float for all OSes that support Metal 3 and later.

Metal provides these type aliases for atomic types:

atomic_int A type alias of atomic<int> for OSes that support Metal 1 and later.

atomic_uint A type alias of atomic<uint> for OSes that support Metal 1 and later.

atomic_bool A type alias of atomic<bool> for OSes that support Metal 2.4 and later.

atomic_ulong A type alias of atomic<ulong> for OSes that support Metal 2.4 and later.

atomic_float A type alias of atomic<float> for OSes that support Metal 3 and later.

Metal atomic functions (as described in section 6.15) can only use Metal atomic data types. These atomic functions are a subset of the C++17 atomic and synchronization functions.

## 2.7 Pixel Data Types

iOS: Metal 2 and later support pixel data types.

macOS: Metal 2.3 and later support pixel data types.

The Metal pixel data type is a templated type that describes the pixel format type and its corresponding ALU type. The header <metal_pixel> defines Metal pixel data. The ALU type represents the type returned by a load operation and the input type specified for a store operation. Pixel data types are generally available in all address spaces. (For more about address spaces, see section 4.)

Table 2.6 lists supported pixel data types in MSL, as well as their size and alignment.

<div align="center">

Table 2.6. Metal pixel data types

</div>

<table border="1"><tr><td>Pixel data type</td><td>Supported values of T</td><td>Size(in bytes)</td><td>Alignment(in bytes)</td></tr><tr><td>r8unorm<T></td><td>half or float</td><td>1</td><td>1</td></tr><tr><td>r8snorm<T></td><td>half or float</td><td>1</td><td>1</td></tr><tr><td>r16unorm<T></td><td>float</td><td>2</td><td>2</td></tr><tr><td>r16snorm<T></td><td>float</td><td>2</td><td>2</td></tr><tr><td>rg8unorm<T></td><td>half2 or float2</td><td>2</td><td>1</td></tr><tr><td>rg8snorm<T></td><td>half2 or float2</td><td>2</td><td>1</td></tr><tr><td>rg16unorm<T></td><td>float2</td><td>4</td><td>2</td></tr><tr><td>rg16snorm<T></td><td>float2</td><td>4</td><td>2</td></tr><tr><td>rgba8unorm<T></td><td>half4 or float4</td><td>4</td><td>1</td></tr><tr><td>srgba8unorm<T></td><td>half4 or float4</td><td>4</td><td>1</td></tr><tr><td>rgba8snorm<T></td><td>half4 or float4</td><td>4</td><td>1</td></tr><tr><td>rgba16unorm<T></td><td>float4</td><td>8</td><td>2</td></tr><tr><td>rgba16snorm<T></td><td>float4</td><td>8</td><td>2</td></tr><tr><td>rgb10a2<T></td><td>half4 or float4</td><td>4</td><td>4</td></tr><tr><td>rg11b10f<T></td><td>half3 or float3</td><td>4</td><td>4</td></tr><tr><td>rgb9e5<T></td><td>half3 or float3</td><td>4</td><td>4</td></tr></table>

Only assignments and equality/inequality comparisons between the pixel data types and their corresponding ALU types are allowed. (The following examples show the buffer(n) attribute, which is explained in section 5.2.1.)

Example:

```c

kernel void

my_kernel(device rgba8unorm<half4> *p [[buffer(0)]],

    uint gid [[thread_position_in_grid]], ...)

{

    rgba8unorm<half4> x = p[index]; half4 val = p[gid];

    ...

    p[gid] = val;

    p[index] = x;

}

```

Example:

```c

struct Foo {

    rgba8unorm<half4> a;

};

kernel void

my_kernel(device Foo *p [[buffer(0)]],

        uint gid [[thread_position_in_grid]], ...)

{

    half4 a = p[gid].a;

    ...

    p[gid].a = a;

}

```

## 2.8 Buffers

MSL implements a buffer as a pointer to a built-in or user defined data type described in the device, constant, or threadgroup address space. (For more about these address space attributes, see sections 4.1, 4.2, and 4.4, respectively.)

Ordinary Metal buffers may contain:

- Basic types such as float and int

- Vector and matrix types

- Arrays of buffer types

- Unions of buffer types

- Structures of buffer types

Note: In Metal 2.3 and later, Metal supports buffers that contain long or ulong data types.

The example below shows buffers as arguments to a function. The first two arguments are buffers in the device address space. The third argument is a buffer in the constant address space.

```cpp

vertex ColorInOut

phong_vertex(const device packed_float3* vertices [[buffer(0)]],

    const device packed_float3* normals [[buffer(1)]],

    constant AAPL::uniforms_t& uniforms [[buffer(2)]],

    unsigned int vid [[vertex_id]])

{

    ...

}

```

For more about the buffer(n) attribute used in the example, see section 5.2.1.

For details about argument buffers, see section 2.13.

## 2.9 Textures

All OS: Metal 3.2 and later support memory_coherence for Apple silicon.

The texture data type is a handle to one-, two-, or three-dimensional texture data that corresponds to all or a portion of a single mipmap level of a texture.

enum class access { sample, read, write, read_write };

In Metal 3.2 and later, texture supports the optional memory coherence parameter (see section 4.8).

enum memory_coherence {

  memory_coherence_threadgroup,

  memory_coherence_device

};

The description below uses the Metal 3.2 template definition with the additional optional coherence parameter. Metal 3.1 and earlier drop that parameter. For example,

// Prior to Metal 3.2

texture1d<T, access a = access::sample>

versus:

// Metal 3.2 and later

texture1d<T, access a = access::sample,

  memory_coherence c = memory_coherence_threadgroup>

The following templates define specific texture data types:

texture1d<T, access a = access::sample,

  memory_coherence c = memory_coherence_threadgroup>

texture1d_array<T, access a = access::sample,

  memory_coherence c = memory_coherence_threadgroup>

texture2d<T, access a = access::sample,

  memory_coherence c = memory_coherence_threadgroup>

texture2d_array<T, access a = access::sample,

  memory_coherence c = memory_coherence_threadgroup>

texture3d<T, access a = access::sample,

  memory_coherence c = memory_coherence_threadgroup>

texturecube<T, access a = access::sample,

  memory_coherence c = memory_coherence_threadgroup>

texturecube_array<T, access a = access::sample,

  memory_coherence c = memory_coherence_threadgroup>

texture2d_ms<T, access a = access::read,

  memory_coherence c = memory_coherence_threadgroup>

texture2d_ms_array<T, access a = access::read,

memory_coherence c = memory_coherence_threadgroup>

To use sample_compare with a depth format, you need to declare one of the following texture types:

depth2d<T, access a = access::sample,

memory_coherence c = memory_coherence_threadgroup>

depth2d_array<T, access a = access::sample,

memory_coherence c = memory_coherence_threadgroup>

depthcube<T, access a = access::sample,

memory_coherence c = memory_coherence_threadgroup>

depthcube_array<T, access a = access::sample,

memory_coherence c = memory_coherence_threadgroup>

macOS supports texture2d_ms_array and depth2d_ms_array in Metal 2 and later. All other types supported in Metal 1 and later.

iOS supports all types except texture2d_ms_array and depth2d_ms_array in Metal 1 and later.

T specifies the color type of one of the components returned when reading from a texture or the color type of one of the components specified when writing to the texture. For texture types (except depth texture types), T can be half, float, short, ushort, int, or uint. For depth texture types, T must be float.

If T is int or short, the data associated with the texture must use a signed integer format. If T is uint or ushort, the data associated with the texture must use an unsigned integer format. If T is half, the data associated with the texture must either be a normalized (signed or unsigned integer) or half-precision format. If T is float, the data associated with the texture must either be a normalized (signed or unsigned integer), half or single-precision format.

These access attributes describe support for accessing a texture:

- sample - A graphics or kernel function can sample the texture object. sample implies the ability to read from a texture with and without a sampler.

- read - Without a sampler, a graphics or kernel function can only read the texture object.

- write - A graphics or kernel function can write to the texture object.

- read_write - A graphics or kernel function can read and write to the texture object.

All OS: Metal 1.2 and later support read_write access. Metal 1 and later support other access qualifiers.

Multisampled textures only support the read attribute. Depth textures only support the sample and read attributes. Sparse textures do not support write or read_write attributes.

The following example uses access qualifiers with texture object arguments:

```cpp

void foo (texture2d<float> imgA [[texture(0)]],

    texture2d<float, access::read> imgB [[texture(1)]],

    texture2d<float, access::write> imgC [[texture(2)]])

{...}

```

(For a description of the texture attribute, see section 5.2.1.)

You can use a texture type as the variable type for any variables declared inside a function. The access attribute for variables of texture type declared inside a function must be access::read or access::sample. Declaring variables inside a function to be a texture type without using access::read or access::sample qualifiers causes a compilation error.

Examples:

```cpp

void foo (texture2d<float> imgA [[texture(0)]],

    texture2d<float, access::read> imgB [[texture(1)]],

    texture2d<float, access::write> imgC [[texture(2)]])

{

    texture2d<float> x = imgA; // OK

    texture2d<float, access::read> y = imgB; // OK

    texture2d<float, access::write> z; // This is illegal.

    ...

}

```

In Metal 3.2 and later, you can indicate whether texture operations are coherent across the device, meaning that texture operations are visible to other threads across thread groups if you synchronize them properly; for example:

```cpp

constant texture2d<float, access::sample,

    memory_coherence_device> gtex [[ texture(2)]];

constant texture2d<int, access::write,

    memory_coherence::memory_coherence_device>

    gtex2 [[ texture(8)]];

```

## 2.9.1 Texture Buffers

See section 4.8 for more information about coherence.

All OS: Metal 2.1 and later support texture buffers.

A texture buffer is a texture type that can access a large 1D array of pixel data and perform dynamic type conversion between pixel formats on that data with optimized performance. Texture buffers handle type conversion more efficiently than other techniques, allowing access to a larger element count, and handling out-of-bounds read access. Similar type conversion can be achieved without texture buffers by either:

- Reading the pixel data (just like any other array) from a texture object and performing the pixel transformation to the desired format.

- Wrapping a texture object around the data of a buffer object, then accessing the shared buffer data via the texture. This wrapping technique provides the pixel conversion, but requires an extra processing step, and the size of the texture is limited.

The following template defines the opaque type texture_buffer, which you can use like any texture:

texture_buffer<T, access a = access::read>

access can be one of read, write, or read_write.

T specifies the type of a component returned when reading from a texture buffer or the type of component specified when writing to a texture buffer. For a texture buffer, T can be one of half, float, short, ushort, int, or uint.

For a format without an alpha channel (such as R, RG, or RGB), an out-of-bounds read returns (0,0,0,1). For a format with alpha (such as RGBA), an out-of-bounds read returns (0, 0,0,0). For some devices, an out-of-bounds read might have a performance penalty.

Metal ignores an out-of-bounds write.

A texture buffer can support more texture data than a generic 1D texture, which has a maximum width of 16384. However, you cannot sample a texture buffer.

A texture buffer also converts data, delivering it in the requested texture format, regardless of the source's format. When creating a texture buffer, you can specify the format of the data in the buffer (for example, RGBA8Unorm), and later the shader function can read it as a converted type (such as float4). As a result, a single pipeline state object can access data stored in different pixel formats without recompilation.

A texture buffer, like a texture type, can be declared as the type of a local variable to a shader function. For information about arrays of texture buffers, see section 2.12.1. For more about texture buffer, see section 6.12.16.

## 2.10 Samplers

The sampler type identifies how to sample a texture. The Metal API allows you to create a sampler object and pass it in an argument to a graphics or kernel function. You can describe a sampler object in the program source instead of in the API. For these cases, you can only specify a subset of the sampler state: the addressing mode, filter mode, normalized coordinates, and comparison function.

Table 2.7 lists the supported sampler state enumerations and their associated values (and defaults). You can specify these states when initializing a sampler in Metal program source.

<div align="center">

Table 2.7. Sampler state enumeration values

</div>

<table border="1"><tr><td>Enumeration</td><td>Valid values</td><td>Description</td></tr><tr><td>coord</td><td>normalized (default) pixel</td><td>When sampling from a texture, specifies whether the texture coordinates are normalized values.</td></tr><tr><td>address</td><td>repeat mirrored_repeat clamp_to_edge (default) clamp_to_zero clamp_to_border</td><td>Sets the addressing mode for all texture coordinates.</td></tr><tr><td>s_address t_address r_address</td><td>repeat mirrored_repeat clamp_to_edge (default) clamp_to_zero clamp_to_border</td><td>Sets the addressing mode for individual texture coordinates.</td></tr><tr><td>border_color macOS: Metal 1.2.iOS: Metal 2.3.</td><td>transparent_black (default) opaque_black opaque_white</td><td>Specifies the border color to use with the clamp_to_border addressing mode.</td></tr><tr><td>filter</td><td>nearest (default) linear</td><td>Sets the magnification and minification filtering modes for texture sampling.</td></tr><tr><td>mag_filter</td><td>nearest (default) linear</td><td>Sets the magnification filtering mode for texture sampling.</td></tr><tr><td>min_filter</td><td>nearest (default) linear</td><td>Sets the minification filtering mode for texture sampling.</td></tr><tr><td>mip_filter</td><td>none (default) nearest linear</td><td>Sets the mipmap filtering mode for texture sampling. If none, the texture is sampled as if it has a single mip level. All samples are read from level 0.</td></tr><tr><td>compare_func</td><td>never (default) less less_equal greater greater_equal equal not_equal always</td><td>Sets the comparison test used by the sample_compare and gather_compare texture functions.</td></tr></table>

<table border="1"><tr><td>Enumeration</td><td>Valid values</td><td>Description</td></tr><tr><td>reduction
All OS: Metal 2.3</td><td>weighted_average
minimum
maximum</td><td>Sets how to compute the filtered pixel value as weighted_average (default), minimum, or maximum.</td></tr><tr><td>bias
All OS: Metal 4.0</td><td>float value</td><td>The level-of-detail(LOD) bias to apply before sampling.See the Metal Feature Set Tables for more information about which GPU families support sampler bias.</td></tr></table>

macOS: Metal 1.2 and later support clamp_to_border address mode and border_color.

iOS: Metal 2.3 and later support clamp_to_border address mode or border_color.

With clamp_to_border, sampling outside a texture only uses the border color for the texture coordinate (and does not use any colors at the edge of the texture). If the address mode is clamp_to_border, then border_color is valid.

clamp_to_zero is equivalent to clamp_to_border with a border color of transparent_black (0.0, 0.0, 0.0) with the alpha component value from the texture. If clamp_to_zero is the address mode for one or more texture coordinates, the other texture coordinates can use an address mode of clamp_to_border if the border color is transparent_black. Otherwise, Metal doesn't define the behavior.

If coord is set to pixel, the min_filter and mag_filter values must be the same, the mip_filter value must be none, and the address modes must be either clamp_to_zero, clamp_to_border, or clamp_to_edge.

In addition to the enumeration types, you can also specify the maximum anisotropic filtering and an level-of-detail (LOD) range for a sampler:

max_anisotropy(int value)

lod_clamp(float min, float max)

The following Metal program source illustrates several ways to declare samplers. (The sampler(n) attribute that appears in the code below is explained in section 5.2.1.) Note that samplers or constant buffers declared in program source do not need these attribute qualifiers. You must use constexpr to declare samplers that you initialize in MSL source:

constexpr sampler s(coord::pixel,

address::clamp_to_zero,

filter::linear);

constexpr sampler a(coord::normalized);

constexpr sampler b(address::repeat);

constexpr sampler s(address::clamp_to_zero,

filter::linear,

compare_func::less);

constexpr sampler s(address::clamp_to_zero,

filter::linear,

compare_func::less,

max_anisotropy(10),

lod_clamp(0.0f, MAXFLOAT));

kernel void

my_kernel(device float4 *p [[buffer(0)]],

    texture2d<float> img [[texture(0)]],
    
    sampler smp [[sampler(3)]],
    
    ...)

{

    ...

}

## 2.11 Imageblocks

iOS: Metal 2 and later support imageblocks.

macOS: Metal 2.3 and later support imageblocks.

An imageblock is a 2D data structure (represented by width, height, and number of samples) allocated in threadgroup memory that is an efficient mechanism for processing 2D image data. Each element of the structure can be a scalar or vector integer or floating-point data type, pixel data types (specified in Table 2.6 in section 2.7), an array of these types, or structures built using these types. The data layout of the imageblock is opaque. You can use an (x, y) coordinate and optionally the sample index to access the elements in the imageblock. The elements in the imageblock associated with a specific (x, y) are the per-thread imageblock data or just the imageblock data.

Section 5.6 details imageblock attributes, including the [[imageblock_data(type)] attribute. Section 6.13 lists the built-in functions for imageblocks.

Imageblocks are only used with fragment and kernel functions. Sections 5.6.3 and 5.6.4 describe how to access an imageblock in a fragment or kernel function, respectively.

For fragment functions, you can access only the fragment's imageblock data (identified by the fragment's pixel position in the tile). Use the tile size to derive the imageblock dimensions.

For kernel functions, all threads in the threadgroup can access the imageblock. You typically derive the imageblock dimensions from the threadgroup size, before you specify the imageblock dimensions.

An imageblock slice refers to a region in the imageblock that describes the values of a given element in the imageblock data structure for all pixel locations or threads in the imageblock. The storage type of the imageblock slice must be compatible with the texture format of the target texture, as listed in Table 2.8.

<div align="center">

Table 2.8. Imageblock slices and compatible target texture formats

</div>

<table border="1"><tr><td>Pixel storage type</td><td>Compatible texture formats</td></tr><tr><td>float,half</td><td>R32Float,R16Float,R8Unorm,R8Snorm,R16Unorm,R16Snorm</td></tr><tr><td>float2,half2</td><td>RG32Float,RG16Float,RG8Unorm,RG8Snorm,RG16Unorm,RG16Snorm</td></tr><tr><td>float4,half4</td><td>RGBA32Float,RGBA16Float,RGBA8Unorm,RGBA8Snorm,RGBA16Unorm,RGBA16Snorm,RGB10A2Unorm,RG11B10Float,RGB9E5Float</td></tr><tr><td>int,short</td><td>R32Sint,R16Sint,R8Sint</td></tr><tr><td>int2,short2</td><td>RG32Sint,RG16Sint,RG8Sint</td></tr><tr><td>int4,short4</td><td>RGBA32Sint,RGBA16Sint,RGBA8Sint</td></tr><tr><td>uint,ushort</td><td>R32Uint,R16Uint,R8Uint</td></tr><tr><td>uint2,ushort2</td><td>RG32Uint,RG16Uint,RG8Uint</td></tr><tr><td>uint4,ushort4</td><td>RGBA32Uint,RGBA16Uint,RGBA8Uint</td></tr><tr><td>r8unorm&lt;T&gt;</td><td>A8Unorm,R8Unorm</td></tr><tr><td>r8snorm&lt;T&gt;</td><td>R8Snorm</td></tr><tr><td>r16unorm&lt;T&gt;</td><td>R16Unorm</td></tr><tr><td>r16snorm&lt;T&gt;</td><td>R16Snorm</td></tr><tr><td>rg8unorm&lt;T&gt;</td><td>RG8Unorm</td></tr><tr><td>rg8snorm&lt;T&gt;</td><td>RG8Snorm</td></tr><tr><td>rg16unorm&lt;T&gt;</td><td>RG16Unorm</td></tr><tr><td>rg16snorm&lt;T&gt;</td><td>RG16Snorm</td></tr><tr><td>rgba8unorm&lt;T&gt;</td><td>RGBA8Unorm,BGRA8Unorm</td></tr><tr><td>srgba8unorm&lt;T&gt;</td><td>RGBA8Unorm_sRGB,BGRA8Unorm_sRGB</td></tr></table>

<table border="1"><tr><td>Pixel storage type</td><td>Compatible texture formats</td></tr><tr><td>rgba8snorm<T></td><td>RGBA8Snorm, BGRA8Unorm</td></tr><tr><td>rgba16unorm<T></td><td>RGBA16Unorm</td></tr><tr><td>rgba16snorm<T></td><td>RGBA16Snorm</td></tr><tr><td>rgb10a2<T></td><td>RGB10A2Unorm</td></tr><tr><td>rg11b10f<T></td><td>RG11B10Float</td></tr><tr><td>rgb9e5<T></td><td>RGB9E5Float</td></tr></table>

## 2.12 Aggregate Types

Metal supports several aggregate types: arrays, structures, classes, and unions.

Do not specify a structure member with an address space attribute, unless the member is a pointer type. All members of an aggregate type must belong to the same address space. (For more about address spaces, see section 4.)

## 2.12.1 Arrays of Textures, Texture Buffers, and Samplers

iOS: Metal 1.2 and later support arrays of textures. Metal 2 and later support arrays of samplers. Metal 2.1 and later support arrays of texture buffers.

macOS: Metal 2 and later support arrays of textures and arrays of samplers. Metal 2.1 and later support arrays of texture buffers.

Declare an array of textures as either:

array<typename T, size_t N>

const array<typename T, size_t N>

typename is a texture type you declare with the access::read or access::sample attribute. Metal 2 and later support an array of writeable textures (access::write) in macOS. Metal 2.2 and later, with Apple GPU Family 5 and later, support it in iOS. (For more about texture types, see section 2.9.)

Construct an array of texture buffers (see section 2.9.1) with the access::read qualifier using:

array<texture_buffer<T>, size_t N>

Declare an array of samplers as either:

array<sampler, size_t N>

const array<sampler, size_t N>

You can pass an array of textures or an array of samplers as an argument to a function (graphics, kernel, or user function) or declare an array of textures or samples as a local variable

inside a function. You can also declare an array of samplers in program scope. Unless used in an argument buffer (see section 2.13), you cannot declare an array<T, N> type (an array of textures, texture buffers, or samplers) in a structure.

MSL also adds support for array_ref<T>. An array_ref<T> represents an immutable array of size() elements of type T. T must be a sampler type or a supported texture type, including texture buffers. The storage for the array is not owned by the array_ref<T> object. Implicit conversions are provided from types with contiguous iterators like metal::array. A common use for array_ref<T> is to pass an array of textures as an argument to functions so they can accept a variety of array types.

The array_ref<T> type cannot be passed as an argument to graphics and kernel functions. However, the array_ref<T> type can be passed as an argument to user functions. The array_ref<T> type cannot be declared as local variables inside functions.

The member functions listed in sections 2.12.1.1 to 2.12.1.3 are available for the array of textures, array of samplers, and the array_ref<T> types.

## 2.12.1.1 Array Element Access with its Operator

Elements of an array of textures, texture buffers, or samplers can be accessed using the [ ] operator:

reference operator[] (size_t pos);

Elements of an array of textures, texture buffers, or samplers, or a templated type array_ref<T> can be accessed using the following variant of the [ ] operator:

constexpr const_reference operator[] (size_t pos) const;

## 2.12.1.2 Array Capacity

size() returns the number of elements in an array of textures, texture buffers, or samplers:

```cpp

constexpr size_t size();

constexpr size_t size() const;

```

## Example:

```cpp

kernel void

my_kernel(const array<texture2d<float>, 10> src [[texture(0)]],

            texture2d<float, access::write> dst [[texture(10)]],

            ...)

{

    for (int i=0; i<src.size(); i++)

    {

        if (is_null_texture(src[i]))

            break;

        process_image(src[i], dst);

    }

}

```

## 2.12.1.3 Constructors for Templated Arrays

constexpr array_ref();

constexpr array_ref(const array_ref &);

array_ref & operator=(const array_ref &);

constexpr array_ref(const T * array, size_t length);

template<size_t N>

constexpr array_ref(const T(&a)[N]);

template<typename T>

constexpr array_ref<T> make_array_ref(const T * array, size_t length)

template<typename T, size_t N>

constexpr array_ref<T> make_array_ref(const T(&a)[N])

Examples of constructing arrays:

```cpp

float4 foo(array_ref<texture2d<float>> src)

{

    float4 clr(0.0f);

    for (int i=0; i<src.size(); i++)

    {

        clr += process_texture(src[i]);

    }

    return clr;

}

kernel void

my_kernel_A(const array<texture2d<float>, 10> src [[texture(0)]],

            texture2d<float, access::write> dst [[texture(10)]],

            ...)

{

    float4 clr = foo(src);

    ...

}

kernel void

my_kernel_B(const array<texture2d<float>, 20> src [[texture(0)]],

            texture2d<float, access::write> dst [[texture(10)]],

            ...)

{

    float4 clr = foo(src);

    ...

}

```

Below is an example of an array of samplers declared in program scope:

```cpp

constexpr array<sampler, 2> samplers = { sampler(address::clamp_to_zero),

    sampler(coord::pixel) };

```

## 2.12.2 Structures of Buffers, Textures, and Samplers

Arguments to a graphics, kernel, visible, or user function can be a structure or a nested structure with members that are buffers, textures, or samplers only. You must pass such a structure by value. Each member of such a structure passed as the argument type to a graphics or kernel function can have an attribute to specify its location (as described in section 5.2.1).

Example of a structure passed as an argument:

```c

struct Foo {

    texture2d<float> a [[texture(0)]];

    depth2d<float> b [[texture(1)]];

};

[[kernel]] void

my_kernel(Foo f)

{...}

```

You can also nest structures, as shown in the following example:

```c

struct Foo {

    texture2d<float> a [[texture(0)]];

    depth2d<float> b [[texture(1)]];

};

struct Bar {

    Foo f;

    sampler s [[sampler(0)]];

};

[[kernel]] void

my_kernel(Bar b)

{...}

```

Below is an example of invalid use-cases that shall result in a compilation error:

```c

struct MyResources {

    texture2d<float> a [[texture(0)]];

    depth2d<float> b [[texture(1)]];

    int c;

};

```

[[kernel]] void

my_kernel(MyResources r) // This is an illegal use.

{...}

## 2.13 Argument Buffers

All OS: Metal 2 and later support argument buffers.

Argument buffers extend the basic buffer types to include pointers (buffers), textures, texture buffers, and samplers. However, argument buffers cannot contain unions. The following example specifies an argument buffer structure called Foo for a function:

```c

struct Foo {

    texture2d<float, access::write> a;

    depth2d<float> b;

    sampler c;

    texture2d<float> d;

    device float4* e;

    texture2d<float> f;

    texture_buffer<float> g;

    int h;

};

kernel void

my_kernel(const Foo & f [[buffer(0)]])

{...}

```

Arrays of textures and samplers can be declared using the existing array<T, N> templated type. Arrays of all other legal buffer types can also be declared using C-style array syntax.

Members of argument buffers can be assigned a generic [[id(n)]] attribute, where n is a 32-bit unsigned integer that can be used to identify the buffer element from the Metal API. Argument buffers can be distinguished from regular buffers if they contain buffers, textures, samplers, or any element with the $ \left[ \left[ \mathrm{i d} \right] \right] $ attribute.

The same index may not be assigned to more than one member of an argument buffer. Manually assigned indices do not need to be contiguous, but they must be monotonically increasing. In the following example, index 0 is automatically assigned to foo1. The [[id(n)]] attribute specifies the index offsets for the t1 and t2 structure members. Since foo2 has no specified index, it is automatically assigned the next index, 4, which is determined by adding 1 to the maximum ID used by the previous structure member.

```c

struct Foo {

    texture2d<float> t1 [[id(1)]];

    texture2d<float> t2 [[id(3)]];

};

struct Bar {

    Foo foo1; // foo1 assigned idx 0, t1 and t2 assigned idx 1 and 3

    Foo foo2; // foo2 assigned idx 4, t1 and t2 assigned idx 5 and 7

};

```

If you omit the [[id]] attribute, Metal automatically assigns an ID according to the following rules:

1. Metal assigns IDs to structure members in order, by adding 1 to the maximum ID of the previous structure member. In the example below, the indices are not provided, so indices 0 and 1 are automatically assigned.

```cpp

struct MaterialTexture {

    texture2d<float> tex; // Assigned index 0

    float4 uvScaleOffset; // Assigned index 1

};

```

2. Metal assigns IDs to array elements in order, by adding 1 to the maximum ID of the previous array element. In the example below, indices 1-3 are automatically assigned to the three array elements of texs1. Indices 4-5 are automatically assigned to the fields in materials[0], indices 6-7 to materials[1], and indices 8-9 to materials[2]. The [[id(20)]] attribute starts by assigning index 20 to constants.

```cpp

struct Material {

    float4 diffuse; // Assigned index 0

    array<texture2d<float>, 3> texs1; // Assigned indices 1-3

    MaterialTexture materials[3]; // Assigned indices 4-9

    int constants [[id(20)]] [4]; // Assigned indices 20-23

};

```

3. If a structure member or array element E is itself a structure or array, Metal assigns indices to its structure members or array elements according to rules 1 and 2 recursively, starting from the ID assigned to E. In the following example, index 4 is explicitly provided for the nested structure called normal, so its elements (previously defined as tex and uvScaleOffset) are assigned IDs 4 and 5, respectively. The elements of the nested structure called specular are assigned IDs 6 and 7 by adding one to the maximum ID (5) used by the previous member.

```c

struct Material {

    MaterialTexture diffuse; // Assigned indices 0, 1

    MaterialTexture normal [[id(4)]]; // Assigned indices 4, 5

    MaterialTexture specular; // Assigned indices 6, 7

}

```

4. Metal assigns IDs to top-level argument buffer arguments starting from 0, according to the previous three rules.

## 2.13.1 Tier 2 Hardware Support for Argument Buffers

With Tier 2 hardware, argument buffers have the following additional capabilities that are not available with Tier 1 hardware.

You can access argument buffers through pointer indexing. This syntax shown below refers to an array of consecutive, independently encoded argument buffers:

```c
kernel void

kern(constant Resources *resArray [[buffer(0)]])

{

    constant Resources &resources = resArray[3];

}

struct TStruct {

    texture2d<float> tex;

};

kernel void

kern(const TStruct *textures [[buffer(0)]]);

```

To support GPU driven pipelines and indirect draw calls and dispatches, you can copy resources between structures and arrays within a function, as shown below:

```cpp

kernel void

copy(constant Foo & src [[buffer(0)]],

    device Foo & dst [[buffer(1)]])

{

    dst.a = src.d;

    ...

}

```

Samplers cannot be copied from the thread address space to the device address space. As a result, samplers can only be copied into an argument buffer directly from another argument buffer. The example below shows both legal and illegal copying:

```cpp

struct Resources {

    sampler sam;

};

kernel void

copy(device Resources *src,

    device Resources *dst,

    sampler sam1)

{

    constexpr sampler sam2;

    dst->sam = src->sam; // Legal: device -> device

    dst->sam = sam1; // Illegal: thread -> device

    dst->sam = sam2; // Illegal: thread -> device

}

```

Argument buffers can contain pointers to other argument buffers:

```c

struct Textures {

    texture2d<float> diffuse;

    texture2d<float> specular;

};

struct Material {

    device Textures *textures;

};

```

fragment float4

fragFunc(device Material & material);

## 2.14 Uniform Type

All OS: Metal 2 and later support uniform types.

## 2.14.1 The Need for a Uniform Type

In the following function example, the variable i is used to index into an array of textures given by texInput. The variable i is nonuniform; that is, it can have a different value for threads executing the graphics or kernel function for a draw or dispatch call, as shown in the example below. Therefore, the texture sampling hardware must handle a sample request that can refer to different textures for threads executing the graphics or kernel function for a draw or dispatch call.

```c

kernel void

my_kernel(array<texture2d<float>, 10> texInput,

        array<texture2d<float>, 10> texOutput,

        sampler s,

        ...

        uint2 gid [[thread_position_in_grid]])

{

    int i = ...;

    float4 color = texInput[i].sample(s, float2(gid));

    ...

    texOutput[i].write(color, float2(gid));

}

```

If the variable i has the same value for all threads (is uniform) executing the graphics or kernel function of a draw or dispatch call and if this information was communicated to the hardware, then the texture sampling hardware can apply appropriate optimizations. A similar argument can be made for texture writes, where a variable computed at runtime is used as an index into an array of textures or to index into one or more buffers.

To indicate that this variable is uniform for all threads executing the graphics or kernel function of a draw or dispatch call, MSL adds a new template class called uniform (available in the header metal_uniform) that can be used for declaring variables inside a graphics or kernel function. This template class can only be instantiated with arithmetic types (such as Boolean, integer, and floating-point) and vector types.

The code below is a modified version of the previous example, where the variable i is declared as a uniform type:

```cpp
kernel void

my_kernel(array<texture2d<float>, 10> texInput,

array<texture2d<float>, 10> texOutput,

sampler s,

...

    uint2 gid [[thread_position_in_grid]])

{

    uniform<int> i = ...;

    float4 color = texInput[i].sample(s, float2(gid));

    ...;

    texOutput[i].write(color, float2(gid));

}

```

## 2.14.2 Behavior of the Uniform Type

If a variable is of the uniform type, and the variable does not have the same value for all threads executing the kernel or graphics function, then the behavior is undefined.

Uniform variables implicitly type convert to nonuniform types. Assigning the result of an expression computed using uniform variables to a uniform variable is legal but assigning a nonuniform variable to a uniform variable results in a compile-time error. In the following example, the multiplication legally converts the uniform variable x into nonuniform product z. However, assigning the nonuniform variable z to the uniform variable b results in a compile-time error.

uniform<int> x = ...;

int y = ...;

int z = x*y; // Here, x converts to a nonuniform for a multiply.

uniform<int> b = z; // Illegal; causes a compile-time error.

To declare an array of uniform elements:

uniform<float> bar[10]; // Elements stored in bar array are uniform.

The uniform type is legal for both parameters and the return type of a function. For example: uniform<int> foo(...); // foo returns a uniform integer value. int bar(uniform<int> a, ...);

It is legal to declare a pointer to a uniform type, but not legal to declare a uniform pointer. For example:

device uniform<int> *ptr; // Values pointed to by ptr are uniform. uniform<device int *> ptr; // Illegal; causes a compile-time error.

The results of expressions that combine uniform with nonuniform variables are nonuniform. If the nonuniform result is assigned to a uniform variable, as in the example below, the behavior is undefined. (The front-end might generate a compile-time error, but it is not guaranteed to do so.)

uniform<int> i = ...;

```c

int j = ...;

if (i < j) { // Nonuniform result for expression (i < j).

    ...

    i++; // Causes a compile-time error, undefined behavior.

}

```

The following example is similar:

bool p = ... // Nonuniform condition.

uniform<int> a = ..., b = ...

uniform<int> c = p ? a : b; // Causes a compile-time error,

// undefined behavior.

## 2.14.3 Uniform Control Flow

When a control flow conditional test is based on a uniform quantity, all program instances follow the same path at that conditional test in a function. Code for control flow based on uniform quantities should be more efficient than code for control flow based on nonuniform quantities.

## 2.15 Visible Function Table

All OS: Metal 2.3 and later support visible function table.

Defined in the header <metal_visible_function_table>, you use the visible_function_table type to represent a table of function pointers to visible functions (see section 5.1.4) that the system stores in device memory. In Metal 2.3 and later, you can use it in a compute (kernel) function. In Metal 2.4 and later, you can use it in fragment, vertex, and tile functions. It is an opaque type, and you can't modify the content of the table from the GPU. You can use a visible_function_table type in an argument buffer or directly pass it to a qualified function using a buffer binding point.

To declare a visible_function_table type with a template parameter T where

T is the signature of the function stored in the table, use the following template function.

visible_function_table<typename T>

The following example shows how to declare a table that is compatible with a function whose definition is "[[visible]] int func(float f)":

visible_function_table<int(float)> functions;

To get a visible function pointer from the table, use the [ ] operator:

using fnptr = T (*)(...) [[visible]]

fnptr operator[](uint index) const;

size() returns the number of function pointer entries in the table: uint size() const

empty() returns true if the table is empty: bool empty() const

The following function can be used to determine if a table is a null visible_function_table. A null visible_function_table is a table that is not pointing to anything.

bool is_null_visible_function_table(visible_function_table<T>);

The following example shows how the table can be passed in a buffer:

```c

using TFuncSig = void(float, int);

kernel void F(uint tid [[thread_position_in_grid]],

              device float* buf [[buffer(0)]],

              visible_function_table<TFuncSig> table [[buffer(1)]])

{

    uint tsize = table.size();

    table[tid % tsize](buf[tid], tid);

}

```

## 2.16 Function Groups Attribute

All OS: Metal 2.3 and later support [[function_groups]].

The optional [[function_groups]] attribute can be used to indicate the possible groups of functions being called from an indirect call through a function pointer or visible_function_table. This is a compiler hint to enable the compiler to optimize the call site. The groups of functions are specified as string literal arguments of the attribute. This attribute can be applied in three different contexts:

- Variable declarations with an initializer expression - It affects all indirect call expressions in the initializer expressions.

- Expression statements — It affects all the indirect call expressions of the given expression.

- Return statements — It affects all the indirect call expressions of the return value expression.

The following examples show how [[function_groups]] can be used:

```cpp

float h(visible_function_table<float(float)> table,

    float (*fnptr[3])(float))

{

    // An indirect call to table[0] is restricted to "group1".

    [[function_groups("group1")]] float x = table[0](1.0f);

```

```cpp

// An indirect call to `fnptr[0]` can call any function.

x += fnptr[0](2.0f);

// An indirect call to `fnptr[1]` is restricted to

// "group2"+"group3".

[[function_groups("group2", "group3")]] return x + fnptr[1](3.0f);

}

```

## 2.17 Ray-Tracing Types

All OS: Metal 2.3 and later support ray-tracing types.

The header <metal_raytracing> defines these types in the namespace metal::raytracing. In Metal 2.3 and later, these types are only supported in a compute function (kernel functions) except where noted below. In Metal 2.4 and later, they are also supported in vertex, fragment, and tile functions. In Metal 3.1 and later, ray tracing supports curves and multilevel instancing.

## 2.17.1 Ray-Tracing Intersection Tags

All OS: Metal 2.3 and later support ray-tracing intersection tags.

The header <metal_raytracing> defines intersection_tags in the namespace metal::raytracing. They are listed in Table 2.9 and are used in ray tracing when defining:

- intersection functions ([[intersection]]; section 5.1.6)

- intersection function tables (intersection_function_table section 2.17.3)

- intersection results (intersection_result section 2.17.4)

- intersector types and associated functions (intersector section 6.18.2)

- acceleration structure types (acceleration_structure section 2.17.7 and 6.18.1)

- intersection queries (intersection_query section 6.18.5).

The tags are used to configure the ray tracing process and control the behavior and semantics of the different types and tables. The tags identify the type of accelerator structure being intersected, the built-in parameters available for intersection functions, the type of intersection function in an intersection function table, the methods available on intersector type or intersection query object, and the data returned in an intersection result type.

The intersection_tags must match in tag type and order between related uses of intersection_function_table, intersection_result, intersector, and intersection_query, or the compiler will generate an error. The acceleration structure type being intersected must match the ordering of instancing, primitive_motion, and instance_motion tags if they are present on the other ray tracing types used to intersect the acceleration structure. When calling intersection functions in an intersection function table, you need to ensure they use the same ordered set of tags, or else the result is undefined.

<div align="center">

Table 2.9. Intersection tags

</div>

<table border="1"><tr><td>Intersection tag</td><td>Description</td></tr><tr><td>instancing</td><td>This tag indicates intersection functions can read the built-in instance_id and/or user_instance_id as described in section 5.2.3.7, and the acceleration structure is an instance acceleration structure.The intersectorintersection_tags...>::intersect() function and intersection_queryintersection_tags...> assume that the acceleration structure needs to be an instance_acceleration_structure and it returns the instance_id value.</td></tr><tr><td>triangle_data</td><td>This tag indicates triangle intersection functions can read input parameters with barycentric_coord or front_facing attribute as described in section 5.2.3.7. This tag cannot be used in defining an acceleration structure.The intersectorintersection_tags...>::intersect() function and intersection_queryintersection_tags...> returns the triangle_barycentric_coord and triangle_front_facing values.</td></tr><tr><td>world_space_data</td><td>This tag indicates intersection functions declared with this tag can query world_space_origin,world_space_direction,object_to_world_transform,andworld_to_object_transformas described in section 5.2.3.7. This tag cannot be used in defining an acceleration structure or intersection_query.It enables support for world space data in intersector and intersection_function_table.</td></tr></table>

<table border="1"><tr><td>Intersection tag</td><td>Description</td></tr><tr><td>primitive_motion
All OS: Metal 2.4 and later</td><td>This tag enables support for primitive level motion in intersector, intersection_function_table, and acceleration structures.</td></tr><tr><td>instance_motion
All OS: Metal 2.4 and later</td><td>This tag enables support for instance level motion in intersector, intersection_function_table, and acceleration structure.</td></tr><tr><td>extended_limits
All OS: Metal 2.4 and later</td><td>This tag indicates acceleration structures passed to intersection functions are built with extended limits for the number of primitives, number of geometries, number of instances, and increases the number of bits used for visibility masks. This tag cannot be used in defining an acceleration structure.</td></tr><tr><td>curve_data
All OS: Metal 3.1 and later</td><td>This tag makes the curve_parameter of the curve intersection point available as a field of intersection_result object from methods of the intersection_query objects, and as input parameter to intersection functions as described in section 5.2.3.7.</td></tr><tr><td>max_levels&lt;Count&gt;
All OS: Metal 3.1 and later</td><td>This tag enables support for multilevel instancing in intersector, intersection_query and intersection_function_table. It cannot be used in acceleration structures. Count is a template parameter that determines the maximum number of acceleration structure levels that can be traversed. It must be between [2,16] for intersection_query. It must be [2,32] for intersector. For intersection_function_table, it needs to match it use with intersection_query or intersector.</td></tr><tr><td>intersection_function_buffer
All OS: Metal 4 and later</td><td>This tag signals that this intersection function is available for use in an intersection function buffer.</td></tr></table>

<table border="1"><tr><td>Intersection tag</td><td>Description</td></tr><tr><td>user_data
All OS: Metal 4 and later</td><td>This tag makes the "user data" pointer available as a parameter marked by user_data_buffer to the function, which is available to pass resources (or any other data) to the intersection function intended for use in an intersection function buffer.</td></tr></table>

In Metal 2.3 and later, the following are valid combinations of intersection tags:

- no tags

- triangle_data

- instancing

- instancing, triangle_data

- instancing, world_space_data

- instancing, triangle_data, world_space_data

Metal 2.4 and later add the following additional valid combinations:

- primitive_motion

- triangle_data, primitive_motion

- instancing, primitive_motion

- instancing, triangle_data, primitive_motion

- instancing, world_space_data, primitive_motion

- instancing, triangle_data, world_space_data, primitive_motion

- instance_motion

- instancing, instance_motion

- instancing, triangle_data, instance_motion

- instancing, world_space_data, instance_motion

- instancing, triangle_data, world_space_data, instance_motion

- instancing, primitive_motion, instance_motion

- instancing, triangle_data, primitive_motion, instance_motion

- instancing, world_space_data, primitive_motion, instance_motion

- instancing, triangle_data, world_space_data, primitive_motion instance_motion

The extended_limits tag may be added to all combinations listed above.

In Metal 3.1 and later, curve_data may be added to all combinations listed above. The intersection tag max_levels<Count> may be added to any combination above containing instancing.

In Metal 4 and later, intersection_function_buffer may be added to all combinations listed above. The tag user_data may only be used in combination with intersection_function_buffer.

## 2.17.2 Ray Type

The ray structure is a container for the properties of the ray required for an intersection.

```c

struct ray

{

    ray(float3 origin = 0.0f, float3 direction = 0.0f,

        float min_distance = 0.0f, float max_distance = INFINITY);

    float3 origin;

    float3 direction;

    float min_distance;

    float max_distance;

};

```

The ray's origin and direction field are in world space. When a ray object is passed into a custom intersection or triangle intersection function, the min_distance and max_distance fields will be based on the current search interval: As candidate intersections are discovered, max_distance will decrease to match the newly narrowed search interval. Within intersection functions, the origin and direction are in object space.

A ray can be invalid. Examples of invalid rays include:

- INFs or NaNs in origin or direction

- min_distance == NaN or max_distance == NaN

- min_distance == INF (Note that max_distance may be positive INF).

- length(ray.direction) == 0.0

- min_distance > max_distance

- min_distance < 0.0 or max_distance < 0.0

The ray direction does not need to be normalized, although it does need to be nonzero.

## 2.17.3 Intersection Function Table

The intersection_function_table<intersection_tags...> structure type describes a table of custom intersection functions passed into the shader as defined from section 5.1.6. The intersection tags are defined from Table 2.9. The intersection tags on intersection_function_table type and the intersection functions must match. An example of such a declaration is:

intersection_function_table<triangle_data, instancing>

intersectionFuncs;

Call the following function to check if the intersection_function_table is null: bool is_null_intersection_function_table( intersection_function_table< intersection_tags...>)

Call the following member function to check if the intersection_function_table is empty:

bool empty() const

Call the following member function to return the number of entries in intersection_function_table:

uint size() const

Metal 3 supports the following function: get_buffer and get_visible_function_table.

Call the following member function to return the buffer at index from the intersection_function_table, where T is a pointer or reference in the device or constant address space:

```cpp

template<typename T>

    T get_buffer(uint index) const

```

Call the following member function to return the visible_function_table<T> at index from the intersection_function_table. T is the signature of the function stored in the table.

```cpp

template <typename T> visible_function_table<T>

get_visible_function_table(uint index) const;

```

Metal 3.1 supports the following functions: set_buffer and set_visible_function_table.

Call the following member functions to set the device or constant buffer object at the index position in the intersection_function_table entry.

void set_buffer(const device void *buf, uint index)

void set_buffer(constant void *buf, uint index)

Call the following member function to set the visible_function_table at the index position in the intersection_function_table, where T is the signature of the function stored in the table.

```cpp

template<typename T>

void set_visible_function_table(visible_function_table<T> vft,

    uint index)

```

## 2.17.4 Intersection Result Type

intersection_result<intersection_tags...> structure where

intersection_tags are defined in Table 2.9. The return structure is defined as:

```cpp

class intersection_type {

  none,

  triangle,

  bounding_box,

  curve // Available in Metal 3.1 and later.

};

template <typename...intersection_tags>

struct intersection_result

{

  intersection_type type;

  float distance;

  uint primitive_id;

  uint geometry_id;

  const device void *primitive_data; // Available in Metal 3 and

                                          // later.

  // Available only if intersection_tags include instancing without

  // max_levels<Count>.

  uint instance_id;

  uint user_instance_id; // Available in Metal 2.4 and

                              // later.

  // In Metal 3.1 and later, replace instance_id and

  // user_instance_id with an array if intersection_tags

  // include instancing and max_levels<Count>.

  uint instance_count; // The number of instances

                              // intersected by the ray.

  uint instance_id[Count - 1]; // The instance IDs of instances

                              // intersected by the ray.

  uint user_instance_id[Count - 1]; // The user instance IDs of

                                  // instances intersected by

                                  // the ray.

    // Available only if intersection_tags include triangle_data.

    float2 triangle_barycentric_coord;

    bool triangle_front_facing;

    // In Metal 2.4 and later, the following is available only if

    // intersection_tags include world_space_data and instancing.

    float4x3 world_to_object_transform;

    float4x3 object_to_world_transform;

    // In Metal 3.1 and later, the following is available only if

    // intersection_tags include curve_data.

    float curve_parameter;

};

```

If a ray is invalid, an intersection::none is returned.

The distance returned is in world space.

For vertex attributes v0, v1, and v2, the attribute value at the specified triangle barycentric point is:

v1 * triangle_barycentric_coord.x +

v2 * triangle_barycentric_coord.y +

v0 * (1.0f - (triangle_barycentric_coord.x +

    triangle_barycentric_coord.y))

## 2.17.5 Intersection Result Reference Type

All OS: Metal 3.2 and later support intersection_result_ref<intersection_tags...> for Apple silicon. The Metal Feature Set Table lists the supported hardware.

In some use cases, it’s possible to avoid a copy of intersection_result by using intersection_result_ref<intersection_tags...> whose lifetime is the duration of the lambda function that passes to the intersector intersect function (see section 6.18.2). The intersection_result_ref<intersection_tags...> structure, where intersection_tags are defined in Table 2.9, is defined as:

```cpp

template <typename...intersection_tags>

struct intersection_result_ref {

public:

    intersection_type get_type() const;

    float get_distance() const;

    uint get_primitive_id() const;

    uint get_geometry_id() const;

    const device void *get_primitive_data() const;

    float3 get_ray_origin() const;

    float3 get_ray_direction() const;

    float get_ray_min_distance() const;

    // Available only if intersection_tags include instancing without.

    // max_levels<Count>.

    uint get_instance_id() const;

    uint get_user_instance_id() const;

    // Available only if intersection_tags include instancing with

    // max_levels<Count>.

    uint get_instance_count() const;

    uint get_instance_id(uint depth) const;

    uint get_user_instance_id(uint depth) const;

    // Available only if intersection_tags include triangle_data.

    float2 get_triangle_barycentric_coord() const;

    bool is_triangle_front_facing() const;

    // Available only if intersection_tags include curve_data.

    float get_curve_parameter() const;

    // Available only if intersection_tags include world_space_data

    // and instancing.

    float4x3 get_object_to_world_transform() const;

    float4x3 get_world_to_object_transform() const;

};

```

## 2.17.6 Intersector Type

The intersector<intersection_tags...> structure type defines an object that controls the acceleration structure traversal and defines functions to intersect rays like intersect(). Use the intersection_tags (described in Table 2.9) when creating the intersector to specialize on which types of acceleration structure it operates on and which functions are available (see section 6.18.2). Intersection tags on the intersector type must match their associated intersection function (section 5.1.6), or the behavior is undefined.

// Create a default intersector.

intersector<> primitiveIntersector;

// Create a specialized intersector to support triangle and

// world space data.

intersector<triangle_data, instancing, world_space_data>

instanceInter;

The intersector<intersection_tags...> struct type provides a convenience type for

the intersection result type defined in section 2.17.6:

intersector<intersection_tags...>::result

## 2.17.7 Acceleration Structure Type

All OS: Metal 2.3 and later support acceleration structure types.

All OS: Metal 2.4 and later support acceleration structure templatized types.

Metal 2.3 and later support two types of acceleration structure:

- primitive_acceleration_structure

- instance_acceleration_structure.

These are opaque objects that can be bound directly using buffer binding points or via argument buffers:

```cpp

struct AccelerationStructs {

    primitive_acceleration_structure prim_accel;

    instance_acceleration_structure inst_accel;

    array<primitive_acceleration_structure, 2> prim_accel_array;

    array<instance_acceleration_structure, 2> inst_accel_array;

};

[[kernel]]

void

intersectInstancesKernel(

    primitive_acceleration_structure prim_accel [[buffer(0)]],

    instance_acceleration_structure inst_accel [[buffer(1)]],

    device AccelerationStructs *accels [[buffer(3)]]) {...}

```

It is possible to create default initialized variables of such types, and the default value is the null value for the acceleration structures.

In Metal 2.4 and later, the acceleration structure is replaced with a templatized type acceleration_structure<intersection_tags...>. The template parameter intersection_tags can be empty or a combination of instancing primitive_motion, or instance_motion as defined in Table 2.9. Intersection tags. For example, the following defines an instance acceleration structure that supports primitive motion:

acceleration_structure<instancing, primitive_motion> accel_struct;

The following combinations of tags can be used to declare a primitive acceleration structure:

- no tags

- primitive_motion

The following combinations of tags can be used to declare an instance acceleration structure:

- instancing

- instancing,primitive_motion

- instancing,instance_motion

- instancing,primitive_motion,instance_motion

To maintain backward compatibility, primitive_acceleration_structure is aliased to acceleration_structure<> and instance_acceleration_structure is aliased to acceleration_structure<instancing>.

As before, these are opaque objects that can be bound directly using buffer binding points or via argument buffers:

```c

struct AccelerationMotionStructs {

    acceleration_structure<primitive_motion> prim_motion_accel;

    acceleration_structure<instancing,

                    instance_motion> inst_motion_accel;

    array<acceleration_structure<>, 2> prim_accel_array;

    array<acceleration_structure<instancing>, 2> inst_accel_array;

};

[[kernel]]

void

intersectMotionKernel(

    acceleration_structure<primitive_motion> prim    [[buffer(15)]],
    
    acceleration_structure<instancing,
    
                    primitive_motion, instance_motion>
    
                    inst    [[buffer(16)]],
    
    device AccelerationMotionStructs        *accels [[buffer(17)]])

{...}

```

When binding these acceleration structures from the Metal API to the compute or graphic functions, the acceleration structure's type must match what is defined in the shader. For instance acceleration structures, you can bind instance acceleration structures without support for primitive_motion to a shader that expects instance acceleration structures with primitive_motion. For example, a Metal buffer with an instance acceleration structure that can be passed to a shader with acceleration_structure<instancing> can also be given to a shader with acceleration_structure<instancing, primitive_motion>. This capability allows you to write one shader function that can handle either an acceleration structure with or without primitive_motion at the cost of the ray tracing runtime checking for primitive motion. To avoid this cost, write two functions where one uses an acceleration structure with primitive_motion and one without.

See section 6.18.1 for the functions to call if the acceleration structure is null.

## 2.17.8 Intersection Query Type

All OS: Metal 2.4 and later support intersection query types.

The intersection_query<intersection_tags...> type defines an object that enables users to fully control the ray tracing process and when to call custom intersection code. The intersection query object provides a set of functions to advance the query through an acceleration structure and query traversal information. Use the intersection_tags (defined in Table 2.9) when creating the intersection_query<intersection_tags...> type to specialize the type of acceleration structure and what functions are available (see section 6.18.5). It supports the following combinations of intersection tags:

- no tags

- triangle_data

- instancing

- instancing, triangle_data

Metal 3.1 supports the following additional combinations:

- instancing, max_levels<Count>

- instancing, triangle_data, max_levels<Count>

In Metal 3.1 and later, curve_data may be added to all combinations listed above.

The intersection_query<intersection_tags...> type has the following restrictions:

- it cannot be used for members of a structure/union

- it cannot be returned from a function

- it cannot be assigned to

These restrictions prevent the intersection query object from being copied.

## 2.18 Interpolant Type

All OS: Metal 2.3 and later support interpolant types.

The interpolant type interpolant<T,P> defined in <metal_interpolate> is a templatized type that encapsulates a fragment shader input for pull-model interpolation (section 6.11). Type parameters T and P represent the input's data type and perspective-correctness, respectively. Supported values for T are the scalar and vector floating-point types. Supported values of P are the types interpolation::perspective and interpolation::no_perspective.

You can declare a variable with the interpolant<T,P> type only in the following contexts:

- As a fragment shader input argument with [['stage_in']]. Such a declaration must match a corresponding vertex shader output argument of type T with the same name or [['user(name)']] attribute. The declaration can't have a sampling-and-interpolation attribute (section 5.4).

- As a local or temporary variable, which needs to be initialized as a copy of the above.

An interpolant<T,P> variable is not automatically convertible to a value of type T. Instead, retrieve a value by calling one of several interpolation methods (see section 6.11). The interpolation shall be perspective-correct if the value of P is interpolation::perspective.

## 2.19 Per-Vertex Values

All OS: Metal 4 and later support per -vertex values.

The vertex value type vertex_value<T> defined in <metal_vertex_value> is a templatized type to provide access to the per-vertex value (preraster per-vertex triangle attributes) in the fragment shader. You can declare a variable with vertex_value<T> as a fragment shader input argument where type T must match the corresponding type in the vertex output.

Call the following function to return the per-vertex value (non-interpolated value) at index i:

```cpp

enum class vertex_index { first, second, third };

T get(vertex_index i);

```

The following example shows a shader that computes the interpolated value as a dot product between the non-interpolated values and the barycentric weights:

```c

struct vertex_in {

    float3 position [[attribute(0)]];

    float4 color [[attribute(1)]];

};

struct vertex_out {

    float4 position [[position]];

    float4 color;

};

[[vertex]] vertex_out vert(vertex_in vert_in [[stage_in]]) { ... }

struct fragment_in {

    float4 position [[position]];

    float3 barycentric_coords [[barycentric_coord,

                                center_no_perspective]];

    vertex_value<float4> color;

};

struct fragment_out {

    float4 color;

};

```

```cpp

[[fragment]] fragment_out frag(fragment_in frag_in [[stage_in]]) {

    fragment_out frag_out;

    auto bc = frag_in.barycentric_coords;

    auto c1 = frag_in.color.get(vertex_index::first);

    auto c2 = frag_in.color.get(vertex_index::second);

    auto c3 = frag_in.color.get(vertex_index::third);

    frag_out.color = c1 * bc.x + c2 * bc.y + c3 * bc.z;

    return frag_out;

}

```

## 2.20 Mesh Shader Types

All OS: Metal 3 and later support mesh shader types. Metal uses these types in the mesh pipeline to render geometry and defines them in the header <metal_mesh>.

## 2.20.1 Mesh Grid Property Type

All OS: Metal 3 and later support mesh grid property types.

An object function (see section 5.1.7) can use the mesh_grid_properties type to specify the size of the mesh grid to dispatch for a given threadgroup from the object stage.

Call the following member function to control the number of threadgroups of the mesh grid that will be dispatched.

void set_threadgroups_per_grid(uint3)

If the member function set_threadgroups_per_grid for a given threadgroup of the object grid is never called, then no mesh grid will be dispatched for the given object grid threadgroup. Calls to set_threadgroups_per_grid behave as a write to threadgroup memory performed by each thread.

## 2.20.2 Mesh Type

All OS: Metal 3 and later support mesh types.

A mesh function (see section 5.1.8) can use an argument of type mesh<V, P, NV, NP, t> structure type to represent the exported mesh data. Table 2.10 describes the mesh template parameters.

<div align="center">

Table 2.10. Mesh template parameter

</div>

<table border="1"><tr><td>Template parameter</td><td>Description</td></tr><tr><td>V</td><td>V is the vertex type.</td></tr><tr><td>P</td><td>P is the primitive type.</td></tr><tr><td>NV</td><td>NV is the maximum number of vertices.</td></tr><tr><td>NP</td><td>NP is the maximum number of primitives.</td></tr><tr><td>t</td><td>t specifies the topology of the mesh. It is one of the following enumeration values:enum topology{point,line,triangle}</td></tr></table>

A valid vertex type v follows the same rules as the vertex function return type defined in section 5.2.3.3 with the following restrictions. The vertex type can be either

- A float4 represents the vertex position

- or a user defined structure:

- Includes a field with the [[position]] attribute.

- May include other fields of scalar or vector of integer or floating-point type.

- Supports the following attributes from Table 2.11. Each attribute can be used once within the vertex type.

<div align="center">

Table 2.11. Mesh vertex attributes

</div>

<table border="1"><tr><td>Attribute</td><td>Corresponding data types</td><td>Description</td></tr><tr><td>clip_distance</td><td>float or float[n]n needs to be known at compile time</td><td>Distance from the vertex to the clipping plane.</td></tr></table>

<table border="1"><tr><td>Attribute</td><td>Corresponding data types</td><td>Description</td></tr><tr><td>invariant</td><td>Not applicable; needs to be used with[[position]]</td><td>Marks the output position such that if the sequence of operations used to compute the output position in multiple vertex shaders is identical, there is a high likelihood that the resulting output position computed by these vertex shaders are the same value. Requires users to pass-fpreserve-invariance.See the description below for more information.</td></tr><tr><td>point_size</td><td>float</td><td>Size of a point primitive.</td></tr><tr><td>position</td><td>float4</td><td>The transformed vertex position.</td></tr><tr><td>shared</td><td>Not applicable</td><td>If present, then for every amplification_id,the output shall have the same value.</td></tr></table>

A valid primitive type follows the same rules as fragment input section 5.2.3.4. A valid primitive type is either:

- void indicating no per-primitive type.

or a user-defined structure:

- Includes fields of scalar or vector of integer or floating-point type.

- Supports only the following attributes from Table 2.12. Each attribute can be used once within the primitive type.

<div align="center">

Table 2.12. Mesh primitive attributes

</div>

<table border="1"><tr><td>Attribute</td><td>Corresponding data types</td><td>Description</td></tr><tr><td>primitive_culled</td><td>bool</td><td>If set to true, the primitive is not rendered.</td></tr><tr><td>primitive_id</td><td>uint</td><td>The per-primitive identifier used with barycentric coordinates.</td></tr></table>

<table border="1"><tr><td>Attribute</td><td>Corresponding data types</td><td>Description</td></tr><tr><td>render_target_array_index</td><td>uchar,ushort,or uint</td><td>The render target array index,which refers to the face of a cubemap,data at a specified depth of a 3D texture,an array slice of a texture array,an array slice,or face of a cubemap array.For a cubemap,the render target array index is the face index,which is a value from0to5.For a cubemap array the render target array index is computed as:array slice index*6+face index.</td></tr><tr><td>viewport_array_index</td><td>uchar,ushort,or uint</td><td>The viewport(and scissor rectangle) index value of the primitive.</td></tr></table>

If the mesh<V, P, NV, NP, t> does not specify a field with [[primitive_culled]] the behavior is the primitive is rendered. If the fragment shader reads the field, the value read is false because that fragment invocation belongs to a nonculled primitive.

Interpolation and sampling qualifiers are accepted on the vertex and primitive type members. The behavior is specified in section 5.2.3.4.

To minimize the possible user errors in mesh-fragment linking, the names of fields for user-defined vertex and primitive type need to be unique between the vertex and primitive type.

An example of mesh<V, P, NV, NP, t> is:

```c

struct VertexOut {

    float4 position [[position]];

};

struct PrimitiveOut

{

    float color [[flat]];

};

using custom_mesh_t = metal::mesh<VertexOut, PrimitiveOut, 64, 64,

                                 metal::topology::triangle>;

```

The mesh types contain the following static data member below.

<div align="center">

Table 2.13. Mesh static members

</div>

<table border="1"><tr><td>Member variable</td><td>Description</td></tr><tr><td>uint max_vertices</td><td>The maximum number of vertices in the mesh(NV).</td></tr></table>

<table border="1"><tr><td>Member variable</td><td>Description</td></tr><tr><td>uint max_primitive</td><td>The maximum number of primitives in the mesh(NP).</td></tr><tr><td>uint indices_per_primitive</td><td>The number of indices per primitive based on topology t.</td></tr><tr><td>uint max_indices</td><td>The maximum number of indices(max_primitive * indices_per_primitive).</td></tr></table>

Call the following member function to set the vertex at index I in the range [0, max_vertices):

void set_vertex(uint I, V v);

If P is not void, call the following member function to set the primitive at index I in the range [0, max_primitive):

```cpp

void set_primitive(uint I, P p);

```

Call the following member to set the primitive count where c is in the range [0, max_primitive]:

```c

void set_primitive_count(uint c);

```

Call the following member to set the index where I is in the range [0, max_indices):

```cpp

void set_index(uint I, uchar v);

```

It is legal to call the following set_indices functions to set the indices if the position in the index buffer is valid and if the position in the index buffer is a multiple of 2 (uchar2 overload) or 2 (uchar4 overload). The index I needs to be in the range [0, max_indices).

```cpp

void set_indices(uint I, uchar2 v);

void set_indices(uint I, uchar4 v);

```

## 2.21 Tensor Types

All OS: Metal 4 and later support tensor types.

Tensors are multidimensional data structures that are fundamental for machine learning. The tensor has:

- data type (all elements are of the same type)

- rank that represents the number of dimensions in the tensor

- layout that represents the extents (size of each dimension) and strides (number of elements to skip past to get to the next element)

Metal defines two types of tensors:

- tensor<...> passed to shaders via arguments, global bindings, argument buffers, or allocated in the shader. Threads can access the storage based on the address space (constant, device, threadgroup, or thread) of the tensor element type.

- cooperative_tensor<...> whose storage is in thread and pre-partitioned among a set of participating threads.

## 2.21.1 Extents Type

The header <metal_tensor> defines the extents type. The type extents <IndexType, size_t... Extents> represents the multidimensional index space of tensors.

<div align="center">

Table 2.14 Extents template parameters

</div>

<table border="1"><tr><td>Template parameter</td><td>Description</td></tr><tr><td>IndexType</td><td>IndexType is the type used for the size of each dimension and for index calculations. It can be any signed or unsigned integer type.</td></tr><tr><td>Extents</td><td>Extents represent the extent (size of an integer interval) for each dimension (rank index). If the extent is determined dynamically (for example, if the size of the dimension is unknown at compile time), use dynamic_extent. Otherwise, the value must be representable in IndexType.</td></tr></table>

<div align="center">

Table 2.15 Extents member types

</div>

<table border="1"><tr><td>Type</td><td>Description</td></tr><tr><td>index_type</td><td>Type used for the size of each dimension and for index calculations based on IndexType.</td></tr></table>

<table border="1"><tr><td>Type</td><td>Description</td></tr><tr><td>size_type</td><td>Type used to describe extents.</td></tr><tr><td>rank_type</td><td>Type used for rank.</td></tr></table>

A convenient alias template dextents<class IndexType, size_t Rank> is provided for extents where Extents for all dimensions is dynamic_extent.

Call the following member function to get the number of dimensions in extents:

static constexpr rank_type rank();

Call the following member function to get the number of dimensions in extents that are dynamic:

```cpp

static constexpr rank_type rank_dynamic();

```

Call the following member function to get the size of an extents at a certain rank index:

static constexpr size_t static_extent(rank_type r);

Call the following member function to get dynamic extent size of an extents at a certain rank index:

constexpr index_type extent(rank_type r);

## 2.21.2 Tensor Type

The header <metal_tensor> defines the tensor<ElementType, Extents, DescriptorType, class... Tag> type. Use this type to pass tensors to shaders via arguments, global bindings, or argument buffers. You can also use this type to create tensors in the shader. Table 2.16 describes the template parameters you can specify when instantiating the template.

<div align="center">

Table 2.16 Tensor template parameters

</div>

<table border="1"><tr><td>Template parameter</td><td>Description</td></tr><tr><td>ElementType</td><td>ElementType is the fully qualified type of the underlying type in the tensor. A fully qualified type includes the value type contained in the tensor, the address space of the underlying storage, and its coherence.
The value type can be one of half,bfloat,float,char,uchar,short,ushort,int,uint,long,or ulong.
The address space is constant,device,threadgroup,or thread(see section4).
The value can be const,volatile,or coherent(device)(see section4.8).</td></tr><tr><td>Extents</td><td>Extents describes the dimensions of the tensor using extents&lt;...&gt;(see section2.21.1).The extent IndexType can be one of short,ushort,int,uint,long,or ulong.</td></tr><tr><td>DescriptorType</td><td>DescriptorType describes where the descriptor lives.It can be either: tensor_handle: tensor contains a handle to the tensor descriptor,or tensor_inline: tensor holds the tensor descriptor.
The default is tensor_handle.</td></tr><tr><td>Tags</td><td>Tags contains the additional compile-time properties.The only supported tag is tensor_offset which you can only use if the DescriptorType is tensor_handle.A tensor marked with that tag holds a set of offsets that shift the origin of the tensor(see section2.21.2.2).</td></tr></table>

<div align="center">

Table 2.17 describes the member types defined by tensor<ElementType, Extents, DescriptorType, Tags...>.

</div>

<div align="center">

Table 2.17 Tensor member type definition

</div>

<table border="1"><tr><td>Type defined</td><td>Description</td></tr><tr><td>element_type</td><td>The fully qualified element type with which you specialized the tensor type.</td></tr><tr><td>value_type</td><td>The unqualified equivalent to element_type.</td></tr><tr><td>extents_type</td><td>The extents type with which you the specialized the tensor type (section 2.21.1).</td></tr><tr><td>index_type</td><td>The type you use for extents, strides, and indices.</td></tr><tr><td>size_type</td><td>The unsigned equivalent of index_type.</td></tr><tr><td>rank_type</td><td>The type you used for the rank of the tensor.</td></tr></table>

All tensors support the following constructors:

```cpp

tensor() thread;

// Copy constructors

tensor(const thread tensor &) thread;

tensor(const device tensor &) thread;

tensor(const device coherent(device) tensor &) thread;

// Conversion constructor extent <-> dextent.

tensor(const thread tensor<element_type,

                OtherExtents,

                tensor_handle,

                Tags...> &other) thread;

tensor(const device tensor<element_type,

                OtherExtents,

                tensor_handle,

                Tags...> &other) thread;

tensor(const device device(coherent)tensor<element_type,

                                OtherExtents,

                                tensor_handle,

                                Tags...> &other) thread;

tensor(const constant tensor<element_type,

                OtherExtents,

                tensor_handle,

                Tags...> &other) thread;

```

```cpp

// Conversion constructor tensor_handle,

// tensor_offset <- tensor_handle.

tensor(const thread tensor<element_type,

                OtherExtents,

                tensor_handle> &other) thread;

tensor(const device tensor<element_type,

                OtherExtents,

                tensor_handle> &other) thread;

tensor(const device device(coherent)

                tensor<element_type,

                OtherExtents,

                tensor_handle> &other) thread;

tensor(const constant tensor<element_type,

                OtherExtents,

                tensor_handle> &other) thread;

```

Call the following member function to get the rank (number of dimensions) of the tensor:

```c

static constexpr size_t get_rank();

```

Call the following member function to get the static extent size (size of a dimension) of the tensor along the $ r^{th} $ dimension:

static constexpr size_t get_static_extent(rank_type r); For example, if extents<int, 32, 64> of the tensor then get_static_extent(0) returns 32 and get_static_extent(1) is 64.

Call the following member function to determine if the extent is static along the $ r^{th} $ dimension:

```cpp

static constexpr bool_has_static_extent(rank_type r);

```

Call the following member function to determine if the tensor has static extents:

```cpp

static constexpr bool has_all_static_extent();

```

Call the following member function to get the extent of the tensor along the $ r^{th} $ dimension:

index_type get_extent(rank_type r);

Call the following member function to get the stride of the tensor along the $ r^{th} $ dimension:

index_type get_stride(rank_type r);

Call the [ ] operator to get a reference to an element of a tensor at multidimensional index. If the index is out of bounds of the tensor, access to the element results in undefined behavior.

```cpp

template<class... OtherIndexTypes>

    reference operator[](OtherIndexTypes...index);

template<class OtherIndexTypes>

    reference operator[]

    thread const array<OtherIndexTypes, get_rank()> &index);

```

Call the following member function to load an element of a tensor at index. The get function supports broadcast semantics where if the multidimension index at i-th is greater than zero and get_extent(i), the effective index is 0. If the effective index is out of bounds of the tensor, the load returns the default value.

```cpp

template<class... OtherIndexTypes>

    value_type get(OtherIndexTypes...index);

template<class OtherIndexTypes>

    value_type get(

        thread const array<OtherIndexTypes, get_rank()> &index);

```

Call the following member function to store a value v to an element of a tensor at index. If the index is out of bounds of the tensor, the GPU drops the store.

```cpp

template<class... OtherIndexTypes>

    void set(value_type v, OtherIndexTypes...index);

template<class OtherIndexTypes>

    void set(value_type, v,

        thread const array<OtherIndexTypes, get_rank()> &index);

```

Call the following member function to get a slice of a tensor whose origin is shifted by index and whose extents are SliceExtents. The returned slice tensor has the same DescriptorType as the original tensor and is either an origin-shifted tensor (see section 2.21.2.2) or a shader-allocated tensor (see section 2.21.2.3). If OtherExtents is dynamic_extent, slice returns the remaining elements starting from index. If this causes the tensor to be out of bounds of the input tensor, it results in undefined behavior.

```cpp

template <size_t... SliceExtents, class... OtherIndexTypes>

    tensor<ElementType, SliceExtents, DescriptorType, SliceTags...>

    slice(OtherIndexTypes... index);

```

See section 2.21.2.2 for some examples.

## 2.21.2.1 Host-bound Tensors

Host-bound tensors are tensors that are allocated and set up on the host. To declare a host-bound tensor, specify tensor_handle to the DescriptorType template parameter. The ElementType may be qualified with either the device or constant address spaces.

```cpp

[[kernel]]

void gemm(tensor<device half, dextents<int, 2>,

           tensor_handle>                ta [[buffer(0)]],

           tensor<constant float, dextents<int, 2>> tb [[buffer(1)]])

{...}

```

The example above defines ta as a tensor allocated in device memory with value type of half. It defines tb as a tensor allocated in constant memory with value type of float. Note that since the default DescriptorType is tensor_handle, it is unnecessary to pass it in this case.

## 2.21.2.2 Origin-shifted Tensors

Origin-shifted tensors are host-bound tensors tagged with tensor_offset. Origin-shifted tensors have their origin shifted by a set of offsets (in number of elements) relative to the base tensor. Calculate the new extents of the tensor relative to the origin, that is, for dimension i:

get_extent(i) = base.get_extent(i) - offset(i);

For example, you can get an origin-shifted tensor using the slice member function of tensor. The return tensor aliases the memory of the base tensor. The first call to slice returns a tensor with dynamic extents because the remaining number of elements in the tensor is based on the original tensor and the shifted origin. The second call returns a 16x16x16 tensor whose origin starts at (32, 32, 32) of the base tensor. The last call returns a 16x16x16 tensor whose origin starts at (16, 16, 32) of the base tensor.

```cpp

[[kernel]]

void offsetTensor(tensor<device float,

                  extents<int, 64, 128, 256>> tbase) {

    // Origin-shifted tensor.

    tensor<device float, dextents<int,3>,

        tensor_handle, tensor_offset> t3 = tbase.slice(8, 16, 32);

    // Origin-shifted 16x16x16 tensor.

    tensor<device float, extents<int, 16, 16, 16>,

        tensor_handle, tensor_offset> t4 =

    tbase.slice<16, 16, 16>(32, 32, 32);

    // Origin-shifted tensor.

    auto t5 = tbase.slice<16, 16, 16>(16, 16, 32);

}

```

## 2.21.2.3 Shader-Allocated Tensors

Shader-allocated (inline) tensors are tensors allocated directly inside a shader. To declare a shader-allocated tensor, specify tensor_inline to the DescriptorType template parameter. You may qualify ElementType with either the device, constant, threadgroup, or thread address spaces. You can't define shader allocated tensor types in an aggregate type (see section 2.12).

Shader-allocated tensors support the following additional constructors:

```cpp

// Raw constructor with pointer, extents, strides.

tensor(data_handle_type ptr,

    thread const OtherExtentsType &_extents,

    thread const array<OtherStrideType, get_rank()>&_strides)

    thread;

// Raw constructor with pointer, extents (with implied packed

// layout for strides).

template <class OtherExtentsType,

tensor(data_handle_type ptr,

    thread const OtherExtentsType &_extents) thread;

```

The example below shows a use of the constructor:

```cpp

[[kernel]] void func1(threadgroup half *buf) {

    tensor<threadgroup half, dextents<int, 3>, tensor_inline>

        t1(buf, dextents<int, 3>(16, 32, 64));

    auto t2 = tensor(buf, dextents<int, 3>(16, 32, 64));

    ...

}

```

## 2.21.3 Cooperative Tensor Type

The header <metal_cooperative_tensor> defines the cooperative_tensor<ElementType, Extents, Layout> type. The cooperative_tensor represents a tensor with elements that are partitioned across a set of participating threads in thread memory. Each thread has access to only the elements in its partition. These threads belong to the same threadgroup and may be spread across consecutive SIMD-groups. You can't define a cooperative_tensor in an aggregate type (see section 2.12).

<div align="center">

Table 2.18 Cooperative tensor template parameters

</div>

<table border="1"><tr><td>Template parameter</td><td>Description</td></tr><tr><td>ElementType</td><td>ElementType is the underlying type in the tensor. For cooperative tensor, the address space is thread.</td></tr><tr><td>Extents</td><td>Extents describes the dimensions of the tensor using extents&lt;...&gt; (see section 2.21.1).</td></tr><tr><td>Layout</td><td>Layout specifies the mapping of the multidimensional coordinate space of the tensor to the prepartitioned storage for each thread.</td></tr></table>

You typically don't construct cooperative_tensor directly as the Layout is device specific. Instead, you use libraries such as Metal Performance Primitives (see section 7), a library of optimized primitives that include operators that work on tensors such as matrix multiplication and convolution. You create them using the tensor operations, which use them to store intermediate results. The tensor operation determines an efficient and performant Layout for a cooperative_tensor based on its usage and the GPU.

## 2.21.3.1 Layout

Layout is an opaque object that provides the following interface that describes the configuration of the cooperative_tensor. The layout is used by the cooperative_tensor to implement its various functions. You don't usually need to call these functions.

Call the following function to return the amount of storage each thread needs to allocate for the cooperative_tensor:

static size_t thread_storage_size();

Call the following function to return the alignment of storage each thread needs to allocate for the cooperative_tensor:

static constexpr size_t thread_storage_align();

Call the following function to return the maximum number of elements that the cooperative_tensor can hold per thread:

static thread_size_type get_capacity(const thread void *this);

Call the following function to determine if the element at idx is valid:

static bool is_valid_element(const thread void *, uint16 idx);

Call the following function to get the pointer to the element at idx. If the idx is invalid, the result is undefined. :

```c

static thread void *

get_element_pointer(const thread void *, uint16_t idx);

```

Call the following function to return the index given the pointer to the element. If the pointer is not a valid element of the cooperative_tensor, the result is undefined.

```c

static uint16_t

get_element_index(const thread void *storage,

                const thread void *element);

```

Call the following function to return the set of multi-dimensional index at idx:

```cpp

template <class OtherIndexType, size_t Rank>

    static array<OtherIndexType, Rank>

    get_multidimensional_index(const thread void *, uint16_t idx);

```

Call the following function to load elements belonging to this thread into per-thread storage:

```cpp

template <class T, class E, class D, class... Tags>

static void load(thread void *storage,

    const thread tensor<T, E, D, Tags...> &);

```

Call the following function to store elements belonging to this thread from per-thread storage into the destination tensor:

```cpp

template <class T, class E, class D, class... Tags>

    static void store(const thread void *storage,

        const thread tensor<T, E, D, Tags...> &);

```

The following function implements this interface when FromIterator can be converted to ToIterator:

```cpp

template <class FromIterator, class ToIterator>

static uint16_t map_index(const thread void *from_storage,

    uint16_t from_idx,

    const thread void *to_storage);

```

<div align="center">

Table 2.19 Cooperative tensor type definition

</div>

<table border="1"><tr><td>Type defined</td><td>Description</td></tr><tr><td>element_type</td><td>The fully qualified element type with which you specialized the cooperative_tensor type.</td></tr><tr><td>value_type</td><td>The unqualified equivalent to element_type.</td></tr><tr><td>extents_type</td><td>The extents type with which you specialized the cooperative_tensor type (section 2.21.1).</td></tr><tr><td>index_type</td><td>The index type you used for extents.</td></tr><tr><td>size_type</td><td>The unsigned equivalent of index_type.</td></tr><tr><td>rank_type</td><td>The type you used for the rank of the cooperative_tensor (via extents).</td></tr><tr><td>thread_index_type</td><td>The index type you used to index per-thread storage.</td></tr><tr><td>thread_size_type</td><td>The unsigned equivalent of thread_index_type.</td></tr><tr><td>data_handle_type</td><td>Pointer to the element_type.</td></tr><tr><td>reference</td><td>Reference to the element_type.</td></tr><tr><td>const_reference</td><td>const equivalent of reference.</td></tr><tr><td>iterator</td><td>Random access iterator to element_type.</td></tr><tr><td>const_iterator</td><td>const equivalent of iterator.</td></tr><tr><td>layout</td><td>The layout with which you specialized the cooperative_tensor type.</td></tr></table>

Call the following member function to get the rank of the cooperative tensor:

```c

static constexpr rank_type get_rank();

```

Call the following member function to cooperatively load all elements from a tensor t into the cooperative tensor. The function supports broadcast semantics where a tensor is expanded into a compatible cooperative tensor. Two tensors are compatible for broadcasting if they have the same rank and when iterating over the dimensions, the sizes are equal or the tensor we are loading from is size 1. For example, you can load a tensor a 64x1 tensor into a 64x2 cooperative tensor.

```cpp

template<class T, class E, calls D, class...>

void load(const thread tensor<T, E, D, ...> &t) thread;

```

Call the following member function to cooperatively store all elements from a cooperative tensor into the tensor t. The function supports broadcast semantics as described in the load. For example, you can store a 64x1 cooperative tensor to a 64x2 tensor.

```cpp

template<class T, class E, class D, class...>

void store(thread tensor<T, E, D, ...> &t) thread const;

```

Call the following member function to the maximum number of elements that are private to this thread. This value is uniform across all threads participating in the cooperative tensor.

thread_size_type get_capacity() thread const;

Call the [ ] operator to get a reference to an element of a cooperative tensor at idx. If the idx is out-of-bound of the cooperative tensor, access to the element results in undefined behavior.

reference operator[](thread_index_type idx);

const_reference operator[](thread_index_type idx) const;

Call the following member function to get the value at it, idx, or ptr from memory owned by this thread:

value_type get(const_iterator it) thread const;

value_type get(thread_index_type idx) thread const;

value_type get(const thread element_type *ptr) thread const;

Call the following member function to set the value at it, idx, or ptr from memory owned by this thread:

```cpp

void set(iterator it, value_type v) thread;

void set(thread_index_type idx, value_type v) thread;

void set(thread element_type *ptr, value_type v) thread;

```

Call the following member function to get the logical multidimensional index that corresponds to the element at it, idx, or ptr:

```cpp

array<index_type, get_rank()>

get_multidimensional_index(const_iterator it) thread const;

array<index_type, get_rank()>

get_multidimensional_index(thread_index_type idx) thread const;

array<index_type, get_rank()>

get_multidimensional_index(

    const thread element_type *ptr) thread const;

```

Call the following member function to determine if the element pointed to by it, idx, or ptr is valid. If the return value is false, the element is invalid, and access to it is undefined behavior.

bool is_valid_element(const_iterator it) const;

bool is_valid_element(thread_index_type idx) const;

bool is_valid_element(const thread element_type *ptr) const;

Call the following member functions to return an iterator to the beginning, which corresponds to the same element at index 0:

iterator begin() thread;

const_iterator begin() thread const;

Call the following member functions to return an iterator to the end:

iterator end() thread;

const_iterator end() thread const;

Call the following member functions to return an iterator corresponding to the element corresponding to idx or ptr:

iterator get_iterator(thread_index_type idx) thread;

const_iterator get_iterator(thread_index_type idx) thread const;

iterator get_iterator(const thread element_type *ptr) thread;

const_iterator get_iterator(

    const thread element_type *ptr) thread const;

Call the following functions that point to the element in this cooperative_tensor that corresponds to the element pointed to by it from another cooperative_tensor. These functions may be exposed if the layout of two cooperative_tensors are compatible.

```cpp

template<class OtherIterator>

    iterator map_iterator(const thread OtherIterator &it);

template<class OtherIterator>

    const_iterator map_iterator(

        const thread OtherIterator &it) const;

```

## 2.22 Type Conversions and Reinterpreting Data

The static_cast operator converts from a scalar or vector type to another scalar or vector type using the default rounding mode with no saturation (when converting to floating-point, round ties to even; when converting to an integer, round toward zero). If the source type is a scalar or vector Boolean, the value false is converted to zero, and the value true is converted to one.

Metal adds an as_type<type-id> operator to allow any scalar or vector data type (that is not a pointer) to be reinterpreted as another scalar or vector data type of the same size. The bits in the operand are returned directly without modification as the new type. The usual type promotion for function arguments is not performed.

For example, as_type<float>(0x3f800000) returns 1.0f, which is the value of the bit pattern 0x3f800000 if viewed as an IEEE-754 single precision value.

Using the as_type<type-id> operator to reinterpret data to a type with a different number of bytes results in an error.

Examples of legal and illegal type conversions:

float f = 1.0f;

// Legal. Contains: 0x3f800000

uint u = as_type<uint>(f);

```cpp

// Legal. Contains:

// (int4)(0x3f800000, 0x40000000, 0x40400000, 0x40800000)

float4 f = float4(1.0f, 2.0f, 3.0f, 4.0f);

int4 i = as_type<int4>(f);

int i;

// Legal.

short2 j = as_type<short2>(i);

half4 f;

// Error. Result and operand have different sizes

float4 g = as_type<float4>(f);

float4 f;

// Legal. g.xyz has same values as f.xyz.

float3 g = as_type<float3>(f);

```

## 2.23 Implicit Type Conversions

Implicit conversions between scalar built-in types (except void) are supported. When an implicit conversion is done, it is not just a re-interpretation of the expression's value but a conversion of that value to an equivalent value in the new type. For example, the integer value 5 is converted to the floating-point value 5.0. A bfloat is an extended floating-point type that only allows implicit conversion to a type of greater floating-point rank. While bfloat can be implicitly converted to float, it cannot be implicitly converted to half, and neither float nor half can be implicitly converted to bfloat.

All vector types are considered to have a higher conversion rank than scalar types. Implicit conversions from a vector type to another vector or scalar type are not permitted and a compilation error results. For example, the following attempt to convert from a 4- component integer vector to a 4- component floating-point vector fails.

```cpp

int4 i;

float4 f = i; // Results in a compile error.

```

Implicit conversions from scalar-to-vector types are supported. The scalar value is replicated in each element of the vector. The scalar may also be subject to the usual arithmetic conversion to the element type used by the vector.

For example:

float4 f = 2.0f; // f = (2.0f, 2.0f, 2.0f, 2.0f)

Implicit conversions from scalar-to-matrix types and vector-to-matrix types are not supported and a compilation error results. Implicit conversions from a matrix type to another matrix, vector or scalar type are not permitted and a compilation error results.

Implicit conversions for pointer types follow the rules described in the C++17 Specification.
