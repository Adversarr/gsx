## 6 Metal Standard Library

This chapter describes functions in the Metal Standard Library (MSLib).

## 6.1 Namespace and Header Files

Metal declares all MSLib functions and enumerations in the metal namespace. In addition to the header files described in the MSLib functions, the <metal_stdlib> header is available and can access all the functions supported by the MSLib.

## 6.2 Common Functions

The header <metal_common> defines the functions in Table 6.1. T is one of the scalar or vector half or float floating-point types.

<div align="center">

Table 6.1. Common functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in common functions</td><td>Description</td></tr><tr><td>T clamp(T x,T minval,T maxval)</td><td>Returns fmin(fmax(x,minval),maxval).Results are undefined if minval&gt;maxval.</td></tr><tr><td>T mix(T x,T y,T a)</td><td>Returns the linear blend of x and y implemented as:x+(y-x)*aor:(1-a)*x+a*ya needs to be a value in the range0.0to1.0.Ifa is not in the range0.0to1.0,the return values are undefined.</td></tr><tr><td>T saturate(T x)</td><td>Clamp the specified value within the range0.0to1.0.</td></tr><tr><td>T sign(T x)</td><td>Returns1.0ifx&gt;0,-0.0ifx=-0.0,+0.0ifx=+0.0,or-1.0ifx&lt;0.Returns0.0ifxis a NaN.</td></tr></table>

<table border="1"><tr><td>Built-in common functions</td><td>Description</td></tr><tr><td>T smoothstep(T edge0,T edge1,T x)</td><td>Returns 0.0 if x&lt;=edge0,and 1.0 if x&gt;=edge1 and performs a smooth Hermite interpolation between0and1whenedge0&lt;x&lt;edge1.This is useful in cases where you want a threshold function with a smooth transitionThis is equivalent to:t=clamp((x-edge0)/(edge1-edge0),0,1);return t*t*(3-2*t);Results are undefined if edge0&gt;=edge1 or ifx,edge0,or edge1 is a NaN.</td></tr><tr><td>T step(T edge,T x)</td><td>Returns 0.0 if x&lt;edge;otherwise,it returns1.0.</td></tr></table>

For single precision floating-point, Metal also supports a precise and fast variant of the following common functions: clamp and saturate. The difference between the Fast and precise function variants handle NaNs differently. In the fast variant, the behavior of NaNs is undefined, whereas the precise variants follow the IEEE 754 rules for NaN handling. The ffast-math compiler option (refer to section 1.6.3) selects the appropriate variant when compiling the Metal source. In addition, the metal::precise and metal::fast nested namespaces provide an explicit way to select the fast or precise variant of these common functions.

## 6.3 Integer Functions

The header <metal_integer> defines the integer functions in Table 6.2. T is one of the scalar or vector integer types. Tu is the corresponding unsigned scalar or vector integer type. T32 is one of the scalar or vector 32-bit int or uint types.

<div align="center">

Table 6.2. Integer functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in integer functions</td><td>Description</td></tr><tr><td>T abs(T x)</td><td>Returns|x|.</td></tr><tr><td>Tu absdiff(T x,T y)</td><td>Returns|x-y|without modulo overflow.</td></tr><tr><td>T addsat(T x,T y)</td><td>Returnsx+yand saturates the result.</td></tr></table>

<table border="1"><tr><td>Built-in integer functions</td><td>Description</td></tr><tr><td>T clamp(T x,T minval,T maxval)</td><td>Returns min(max(x,minval),maxval).Results are undefined if minval&gt;maxval.</td></tr><tr><td>T clz(T x)</td><td>Returns the number of leading0-bits inx,starting at the most significant bit position.Ifx is0,returns the size in bits of the type ofx or component type ofx,ifx is a vector</td></tr><tr><td>T ctz(T x)</td><td>Returns the count of trailing0-bits inx.Ifx is0,returns the size in bits of the type ofx or ifx is a vector,the component type ofx.</td></tr><tr><td>T extract_bits(T x,uint offset,uint bits)All OS:Metal1.2 and later</td><td>Extract bits[offset,offset+bits-1]fromx,returning them in the least significant bits of the result.For unsigned data types,the most significant bits of the result are set to zero.For signed data types,the most significant bits are set to the value ofbit offset+bits-1.Ifbits is zero,the result is zero.If the sum ofoffset andbits is greater than the number ofbits used to store the operand,the result is undefined.</td></tr><tr><td>T hadd(T x,T y)</td><td>Returns(x+y)>>1.The intermediate sum does not modulo overflow.</td></tr><tr><td>T insert_bits(T base,T insert,uint offset,uint bits)All OS:Metal1.2 and later</td><td>Returns the insertion of thebits least-significant bits ofinsert intobase.The result has bits[offset,offset+bits-1]taken frombits[0,bits-1]ofinsert,and all other bits are taken directly from the corresponding bits ofbase.Ifbits is zero,the result isbase.If the sum ofoffset andbits is greater than the number ofbits used to store the operand,the result is undefined.</td></tr><tr><td>T32 mad24(T32x,T32y,T32z)All OS:Metal2.1 and later</td><td>Usesmul24to multiply two24-bit integer valuesx andy,adds the32-bit integer result to the32-bit integerz,and returns that sum.</td></tr><tr><td>T madhi(T a,T b,T c)</td><td>Returnsmulhi(a,b)+c.</td></tr><tr><td>T madsat(T a,T b,T c)</td><td>Returnsa*b+cand saturates the result.</td></tr></table>

<table border="1"><tr><td>Built-in integer functions</td><td>Description</td></tr><tr><td>T max(T x,T y)</td><td>Returns y if x<y,otherwise it returns x.</td></tr><tr><td>T max3(T x,T y,T z)
All OS:Metal2.1 and later</td><td>Returns max(x,max(y,z)).</td></tr><tr><td>T median3(T x,T y,T z)
All OS:Metal2.1 and later</td><td>Return the middle value of x,y,and z.</td></tr><tr><td>T min(T x,T y)</td><td>Returns y if y<x,otherwise,it returns x.</td></tr><tr><td>T min3(T x,T y,T z)
All OS:Metal2.1 and later</td><td>Returns min(x,min(y,z)).</td></tr><tr><td>T32 mul24(T32 x,T32 y)
All OS:Metal2.1 and later</td><td>Multiplies two 24-bit integer values x and y and returns the 32-bit integer result.x and y are 32-bit integers but only the low 24 bits perform the multiplication.(See details following this table.)</td></tr><tr><td>T mulhi(T x,T y)</td><td>Computes x*y and returns the high half of the product of x and y.</td></tr><tr><td>T popcount(T x)</td><td>Returns the number of nonzero bits in x.</td></tr><tr><td>T reverse_bits(T x)
All OS:Metal2.1 and later</td><td>Returns the reversal of the bits of x.The bit numbered n of the result is taken from bit(bits-1)-n of x,where bits is the total number of bits used to represent x.</td></tr><tr><td>T rhadd(T x,T y)</td><td>Returns(x+y+1)>>1.The intermediate sum does not modulo overflow.</td></tr><tr><td>T rotate(T v,T i)</td><td>For each element in v,the bits are shifted left by the number of bits given by the corresponding element in i.Bits shifted off the left side of the element are shifted back in from the right.</td></tr><tr><td>T subsat(T x,T y)</td><td>Returns x-y and saturates the result.</td></tr></table>

The mul24 function only operates as described if x and y are signed integers and x and y are in the range $[-2^23, 2^23 - 1]$, or if x and y are unsigned integers and x and y are in the range $[0, 2^24 - 1]$. If x and y are not in this range, the multiplication result is implementation-defined.

## 6.4 Relational Functions

The header <metal_relational> defines the relational functions in Table 6.3. T is one of the scalar or vector floating-point types including bfloat types. Ti is one of the scalar or vector integer or Boolean types. Tb only refers to the scalar or vector Boolean types.

<div align="center">

Table 6.3. Relational functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in relational functions</td><td>Description</td></tr><tr><td>bool all(Tb x)</td><td>Returns true only if all components of x are true.</td></tr><tr><td>bool any(Tb x)</td><td>Returns true only if any component of x are true.</td></tr><tr><td>Tb isfinite(T x)</td><td>Test for finite value.</td></tr><tr><td>Tb isinf(T x)</td><td>Test for infinity value (positive or negative).</td></tr><tr><td>Tb isnan(T x)</td><td>Test for a NaN.</td></tr><tr><td>Tb isnormal(T x)</td><td>Test for a normal value.</td></tr><tr><td>Tb isordered(T x, Ty)</td><td>Test if arguments are ordered.isordered() takes arguments x and y and returns the result(x==x)&&(y==y).</td></tr><tr><td>Tb isunordered(T x, Ty)</td><td>Test if arguments are unordered.isunordered() takes arguments x and y and returns true if x or y is NaN; otherwise, returns false.</td></tr><tr><td>Tb not(Tb x)</td><td>Returns the componentwise logical complement of x.</td></tr><tr><td>T select(Ta,Tb,Tb c)
Ti select(Ti a,
Ti b,
Tb c)</td><td>For each component of a vector type,
result[i]=c[i]?b[i]:a[i]
For a scalar type,
result=c?b:a</td></tr><tr><td>Tb signbit(T x)</td><td>Test for sign bit. Returns true if the sign bit is set for the floating-point value in x; otherwise, returns false.</td></tr></table>

## 6.5 Math Functions

The header <metal_math> defines the math functions in Table 6.4. T is one of the scalar or vector half or float floating-point types. Ti refers only to the scalar or vector integer types.

<div align="center">

Table 6.4. Math functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in math functions</td><td>Description</td></tr><tr><td>T acos(T x)</td><td>Compute arc cosine of x.</td></tr><tr><td>T acosh(T x)</td><td>Compute inverse hyperbolic cosine of x.</td></tr><tr><td>T asin(T x)</td><td>Compute arc sine function of x.</td></tr></table>

<table border="1"><tr><td>Built-in math functions</td><td>Description</td></tr><tr><td>T asinh(T x)</td><td>Compute inverse hyperbolic sine of x.</td></tr><tr><td>T atan(T y_over_x)</td><td>Compute arc tangent of x.</td></tr><tr><td>T atan2(T y,T x)</td><td>Compute arc tangent of y over x.</td></tr><tr><td>T atanh(T x)</td><td>Compute hyperbolic arc tangent of x.</td></tr><tr><td>T ceil(T x)</td><td>Round x to integral value using the round to positive infinity rounding mode.</td></tr><tr><td>T copysign(T x,T y)</td><td>Return x with its sign changed to match the sign of y.</td></tr><tr><td>T cos(T x)</td><td>Compute cosine of x.</td></tr><tr><td>T cosh(T x)</td><td>Compute hyperbolic cosine of x.</td></tr><tr><td>T cospi(T x)</td><td>Compute cos(πx).</td></tr><tr><td>T divide(T x,T y)</td><td>Compute x/y.</td></tr><tr><td>T exp(T x)</td><td>Exponential base e function.</td></tr><tr><td>T exp2(T x)</td><td>Exponential base 2 function.</td></tr><tr><td>T exp10(T x)</td><td>Exponential base 10 function.</td></tr><tr><td>T fabs(T x)
T abs(T x)</td><td>Compute absolute value of a floating-point number.</td></tr><tr><td>T fdim(T x,T y)</td><td>x-y if x&gt;y;+0 if x&lt;=y.</td></tr><tr><td>T floor(T x)</td><td>Round x to integral value using the round to negative infinity rounding mode.</td></tr><tr><td>T fma(T a,T b,T c)</td><td>Returns the correctly rounded floating-point representation of the sum of c with the infinitely precise product of a and b.Rounding of intermediate products shall not occur. Edge case behavior is per the IEEE 754-2008 standard.</td></tr><tr><td>T fmax(T x,T y)
T max(T x,T y)</td><td>Returns y if x&lt;y, otherwise returns x.If one argument is a NaN,fmax() returns the other argument.If both arguments are NaNs,fmax() returns a NaN.If x and y are denormals and the GPU doesn&#x27;t support denormals,either value may be returned.</td></tr></table>

<table border="1"><tr><td>Built-in math functions</td><td>Description</td></tr><tr><td>T fmax3(T x,T y,T z)
T max3(T x,T y,T z)
All OS: Metal 2.1 and later</td><td>Returns fmax(x,fmax(y,z)).</td></tr><tr><td>T fmedian3(T x,T y,T z)
All OS: Metal 1 and later
T median3(T x,T y,T z)
All OS: Metal 2.1 and later</td><td>Returns the middle value of x,y,and z.(If one or more values are NaN,see discussion after this table.)</td></tr><tr><td>T fmin(T x,T y)
T min(T x,T y)</td><td>Returns y if y&lt;x,otherwise it returns x.If one argument is a NaN,fmin() returns the other argument.If both arguments are NaNs,fmin() returns a NaN.If x and y are denormals and the GPU doesn&#x27;t support denormals,either value may be returned.</td></tr><tr><td>T fmin3(T x,T y,T z)
T min3(T x,T y,T z)
All OS: Metal 2.1 and later</td><td>Returns fmin(x,fmin(y,z)).</td></tr><tr><td>T fmod(T x,T y)</td><td>Returns x-y*trunc(x/y).</td></tr><tr><td>T fract(T x)</td><td>Returns the fractional part of x that is greater than or equal to0 or less than1.</td></tr><tr><td>T frexp(T x,Ti &amp; exponent)</td><td>Extract mantissa and exponent from x.For each component the mantissa returned is a float with magnitude in the interval[1/2,1) or0.Each component of x equals mantissa returned *2exp.</td></tr><tr><td>Ti ilogb(T x)</td><td>Return the exponent as an integer value.</td></tr><tr><td>T ldexp(T x,Ti k)</td><td>Multiply x by2 to the power k.</td></tr><tr><td>T log(T x)</td><td>Compute the natural logarithm of x.</td></tr><tr><td>T log2(T x)</td><td>Compute the base2 logarithm of x.</td></tr><tr><td>T log10(T x)</td><td>Compute the base10 logarithm of x.</td></tr><tr><td>T modf(T x,T &amp; intval)</td><td>Decompose a floating-point number.The modf function breaks the argument x into integral and fractional parts,each of which has the same sign as the argument.Returns the fractional value.The integral value is returned in intval.</td></tr><tr><td>T nextafter(T x,T y)
All OS: Metal 3.1 and later</td><td>Return next representable floating-point value after x in the direction of y.If x equals y,return</td></tr></table>

<table border="1"><tr><td>Built-in math functions</td><td>Description</td></tr><tr><td></td><td>y. Note that if both x and y represent the floating-point zero values, the result has sign of y. If either x or y is NaN, return NaN.</td></tr><tr><td>T pow(T x,T y)</td><td>Compute x to the power y.</td></tr><tr><td>T powr(T x,T y)</td><td>Compute x to the power y, where x is $>=0$.</td></tr><tr><td>T rint(T x)</td><td>Round x to integral value using round ties to even rounding mode in floating-point format.</td></tr><tr><td>T round(T x)</td><td>Return the integral value nearest to x, rounding halfway cases away from zero.</td></tr><tr><td>T rsqrt(T x)</td><td>Compute inverse square root of x.</td></tr><tr><td>T sin(T x)</td><td>Compute sine of x.</td></tr><tr><td>T sincos(T x,T cosval)</td><td>Compute sine and cosine of x. Return the computed sine in the function return value, and return the computed cosine in cosval.</td></tr><tr><td>T sinh(T x)</td><td>Compute hyperbolic sine of x.</td></tr><tr><td>T sinpi(T x)</td><td>Compute sin(πx).</td></tr><tr><td>T sqrt(T x)</td><td>Compute square root of x.</td></tr><tr><td>T tan(T x)</td><td>Compute tangent of x.</td></tr><tr><td>T tanh(T x)</td><td>Compute hyperbolic tangent of x.</td></tr><tr><td>T tanpi(T x)</td><td>Compute tan(πx).</td></tr><tr><td>T trunc(T x)</td><td>Round x to integral value using the round toward zero rounding mode.</td></tr></table>

For fmedian3, if all values are NaN, return NaN. Otherwise, treat NaN as missing data and remove it from the set. If two values are NaN, return the non-NaN value. If one of the values is NaN, the function can return either non-NaN value.

For single precision floating-point, Metal supports two variants for most of the math functions listed in Table 6.4: the precise and the fast variants. See Table 8.2 in section 8.4 for the list of fast math functions and their precision. The ffast-math compiler option (refer to section 1.6.3) selects the appropriate variant when compiling the Metal source. In addition, the metal::precise and metal::fast nested namespaces provide an explicit way to select the fast or precise variant of these math functions for single precision floating-point.

Examples:

float x;

float a = sin(x); // Use fast or precise version of sin based on // whether you specify -ffast-math as // compile option.

float b = fast::sin(x); // Use fast version of sin().

float c = precise::cos(x); // Use precise version of cos().

All OS: Metal 1.2 and later support the constants in Table 6.5 and Table 6.6.

Table 6.5 lists available symbolic constants with values of type float that are accurate within the precision of a single-precision floating-point number.

<div align="center">

Table 6.5. Constants for single-precision floating-point math functions

</div>

<table border="1"><tr><td>Constant name</td><td>Description</td></tr><tr><td>MAXFLOAT</td><td>Value of maximum noninfinite single precision floating-point number.</td></tr><tr><td>HUGE_VALF</td><td>A positive float constant expression.HUGE_VALF evaluates to +infinity.</td></tr><tr><td>INFINITY</td><td>A constant expression of type float representing positive or unsigned infinity.</td></tr><tr><td>NAN</td><td>A constant expression of type float representing a quiet NaN.</td></tr><tr><td>M_E_F</td><td>Value of e</td></tr><tr><td>M_LOG2E_F</td><td>Value of log2e</td></tr><tr><td>M_LOG10E_F</td><td>Value of log10e</td></tr><tr><td>M_LN2_F</td><td>Value of loge2</td></tr><tr><td>M_LN10_F</td><td>Value of loge10</td></tr><tr><td>M_PI_F</td><td>Value of π</td></tr><tr><td>M_PI_2_F</td><td>Value of π/2</td></tr><tr><td>M_PI_4_F</td><td>Value of π/4</td></tr><tr><td>M_1_PI_F</td><td>Value of 1/π</td></tr><tr><td>M_2_PI_F</td><td>Value of 2/π</td></tr><tr><td>M_2_SQRTPI_F</td><td>Value of 2/√π</td></tr><tr><td>M_SQRT2_F</td><td>Value of √2</td></tr><tr><td>M_SQRT1_2_F</td><td>Value of 1/√2</td></tr></table>

Table 6.6 lists available symbolic constants with values of type half that are accurate within the precision of a half-precision floating-point number.

<div align="center">

Table 6.6. Constants for half-precision floating-point math functions

</div>

<table border="1"><tr><td>Constant name</td><td>Description</td></tr><tr><td>MAXHALF</td><td>Value of maximum noninfinite half precision floating-point number.</td></tr><tr><td>HUGE_VALH</td><td>A positive half constant expression. HUGE_VALH evaluates to +infinity.</td></tr><tr><td>M_E_H</td><td>Value of e</td></tr><tr><td>M_LOG2E_H</td><td>Value of log2e</td></tr><tr><td>M_LOG10E_H</td><td>Value of log10e</td></tr><tr><td>M_LN2_H</td><td>Value of loge2</td></tr><tr><td>M_LN10_H</td><td>Value of loge10</td></tr><tr><td>M_PI_H</td><td>Value of π</td></tr><tr><td>M_PI_2_H</td><td>Value of π/2</td></tr><tr><td>M_PI_4_H</td><td>Value of π/4</td></tr><tr><td>M_1_PI_H</td><td>Value of 1/π</td></tr><tr><td>M_2_PI_H</td><td>Value of 2/π</td></tr><tr><td>M_2_SQRTPI_H</td><td>Value of 2/√π</td></tr><tr><td>M_SQRT2_H</td><td>Value of √2</td></tr><tr><td>M_SQRT1_2_H</td><td>Value of 1/√2</td></tr></table>

<div align="center">

Table 6.7 lists available symbolic constants with values of type bfloat that are accurate within the precision of a bfloat floating-point number.

</div>

<div align="center">

Table 6.7. Constants for bfloat floating-point math functions

</div>

<table border="1"><tr><td>Constant name</td><td>Description</td></tr><tr><td>MAXBFLOAT</td><td>Value of maximum noninfinite bfloat floating-point number.</td></tr></table>

<table border="1"><tr><td>Constant name</td><td>Description</td></tr><tr><td>HUGE_VALBF</td><td>A positive half constant expression.HUGE_VALBF evaluates to+infinity.</td></tr><tr><td>M_E_BF</td><td>Value ofe</td></tr><tr><td>M_LOG2E_BF</td><td>Value oflog2e</td></tr><tr><td>M_LOG10E_BF</td><td>Value oflog10e</td></tr><tr><td>M_LN2_BF</td><td>Value ofloge2</td></tr><tr><td>M_LN10_BF</td><td>Value ofloge10</td></tr><tr><td>M_PI_BF</td><td>Value ofπ</td></tr><tr><td>M_PI_2_BF</td><td>Value ofπ/2</td></tr><tr><td>M_PI_4_BF</td><td>Value ofπ/4</td></tr><tr><td>M_1_PI_BF</td><td>Value of1/π</td></tr><tr><td>M_2_PI_BF</td><td>Value of2/π</td></tr><tr><td>M_2_SQRTPI_BF</td><td>Value of2/√π</td></tr><tr><td>M_SQRT2_BF</td><td>Value of√2</td></tr><tr><td>M_SQRT1_2_BF</td><td>Value of1/√2</td></tr></table>

## 6.6 Matrix Functions

The header <metal_matrix> defines the functions in Table 6.8. T is float or half.

<div align="center">

Table 6.8. Matrix functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in matrix functions</td><td>Description</td></tr><tr><td>float determinant(floatnxn) half determinant(halfnxn)</td><td>Compute the determinant of the matrix. The matrix needs to be a square matrix.</td></tr><tr><td>floatmxn transpose(floatnxm) halfmxn transpose(halfnxm)</td><td>Transpose a matrix.</td></tr></table>

Example:

float4x4 mA;

float det = determinant(mA);

## 6.7 SIMD-Group Matrix Functions

The header <metal_simdgroup_matrix> defines the SIMD-group Matrix functions.

## 6.7.1 Creating, Loading, and Storing Matrix Elements

Metal Shading Library supports the following functions to initialize a SIMD-group matrix with a value, load data from threadgroup or device memory, and store data to threadgroup or device memory.

<div align="center">

Table 6.9. SIMD-Group matrix load and stores

</div>

<table border="1"><tr><td>Functions</td><td>Description</td></tr><tr><td>simdgroup_matrix&lt;T,Cols,Rows&gt;(T dval)</td><td>Creates a diagonal matrix with the given value.</td></tr><tr><td>simdgroup_matrix&lt;T,Cols,Rows&gt;make_filled_simdgroup_matrix(T value)</td><td>Initializes a SIMD-group matrix filled with the given value.</td></tr><tr><td>void simdgroup_load(thread simdgroup_matrix&lt;T,Cols,Rows&gt;&amp;d,const threadgroup T *src,ulong elements_per_row=Cols,ulong2 matrix_origin=0,bool transpose_matrix=false)</td><td>Loads data from threadgroup memory into a SIMD-group matrix.The elements_per_row parameter indicates the number of elements in the source memory layout.</td></tr><tr><td>void simdgroup_load(thread simdgroup_matrix&lt;T,Cols,Rows&gt;&amp;d,const device T *src,ulong elements_per_row=Cols,ulong2 matrix_origin=0,bool transpose_matrix=false)</td><td>Loads data from device memory into a SIMD-group matrix.The elements_per_row parameter indicates the number of elements in the source memory layout.</td></tr><tr><td>void simdgroup_store(thread simdgroup_matrix&lt;T,Cols,Rows&gt;a,threadgroup T *dst,ulong elements_per_row=Cols,ulong2 matrix_origin=0,bool transpose_matrix=false)</td><td>Stores data from a SIMD-group matrix into threadgroup memory.The elements_per_row parameter indicates the number of elements in the destination memory layout.</td></tr></table>

<table border="1"><tr><td>Functions</td><td>Description</td></tr><tr><td>void simdgroup_store( thread simdgroup_matrix<T,Cols,Rows&gt; a, device T *dst, ulong elements_per_row = Cols, ulong2 matrix_origin = 0, bool transpose_matrix = false)</td><td>Stores data from a SIMD-group matrix into device memory. The elements_per_row parameter indicates the number of elements in the destination memory layout.</td></tr></table>

## 6.7.2 Matrix Operations

SIMD-group matrices support multiply-accumulate and multiple operations.

<div align="center">

Table 6.10. SIMD-Group operations

</div>

<table border="1"><tr><td>Operations</td><td>Description</td></tr><tr><td>void simdgroup_multiply_accumulate( thread simdgroup_matrix&lt;T,Cols,Rows&gt;&amp;d, thread simdgroup_matrix&lt;T,K,Rows&gt;&amp;a, thread simdgroup_matrix&lt;T,Cols,K&gt;&amp;b, thread simdgroup_matrix&lt;T,Cols,Rows&gt;&amp;c)</td><td>Returns d=a*b+c</td></tr><tr><td>void simdgroup_multiply( thread simdgroup_matrix&lt;T,Cols,Rows&gt;&amp;d, thread simdgroup_matrix&lt;T,K,Rows&gt;&amp;a, thread simdgroup_matrix&lt;T,Cols,K&gt;&amp;b)</td><td>Returns d=a*b</td></tr><tr><td>$\cdot$</td><td>Returns a*b</td></tr></table>

Here is an example of how to use SIMD-group matrices:

```cpp

kernel void float_matmad(device float *pMatA, device float *pMatB,

                         device float *pMatC, device float *pMatR)

{

    simdgroup_float8x8 sgMatA;

    simdgroup_float8x8 sgMatB;

    simdgroup_float8x8 sgMatC;

    simdgroup_float8x8 sgMatR;

    simdgroup_load(sgMatA, pMatA);

    simdgroup_load(sgMatB, pMatB);

    simdgroup_load(sgMatC, pMatC);

    simdgroup_multiply_accumulate(sgMatR, sgMatA, sgMatB, sgMatC);

    simdgroup_store(sgMatR, pMatR);

}

```

## 6.8 Geometric Functions

The functions in Table 6.11 are defined in the header <metal_geometric>. T is a vector floating-point type (floatn or halfn). Ts refers to the corresponding scalar type. (If T is floatn, the scalar type Ts is float. If T is halfn, Ts is half.)

<div align="center">

Table 6.11. Geometric functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in geometric functions</td><td>Description</td></tr><tr><td>T cross(T x,T y)</td><td>Return the cross product of x and y.T needs to be a 3-component vector type.</td></tr><tr><td>Ts distance(T x,T y)</td><td>Return the distance between x and y,which is length(x-y)</td></tr><tr><td>Ts distance_squared(T x,T y)</td><td>Return the square of the distance between x and y.</td></tr><tr><td>Ts dot(T x,T y)</td><td>Return the dot product of x and y,which is x[0]*y[0]+x[1]*y[1]+...</td></tr><tr><td>T faceforward(T N,T I,T Nref)</td><td>If dot(Nref,I)&lt;0.0 return N,otherwise return-N.</td></tr><tr><td>Ts length(T x)</td><td>Return the length of vector x,which is sqrt(x[0]2+x[1]2+...</td></tr><tr><td>Ts length_squared(T x)</td><td>Return the square of the length of vector x,which is(x[0]2+x[1]2+...</td></tr><tr><td>T normalize(T x)</td><td>Return a vector in the same direction as x but with a length of 1.</td></tr><tr><td>T reflect(T I,T N)</td><td>For the incident vector I and surface orientation N,compute normalized N(NN),and return the reflection direction:I-2*dot(NN,I)*NN.</td></tr><tr><td>T refract(T I,T N,Ts eta)</td><td>For the incident vector I and surface normal N,and the ratio of indices of refraction eta,return the refraction vector.The input parameters for the incident vector I and the surface normal N needs to already be normalized to get the desired results.</td></tr></table>

For single precision floating-point, Metal also supports a precise and fast variant of the following geometric functions: distance, length, and normalize. To select the appropriate variant when compiling the Metal source, use the ffast-math compiler option (refer to section 1.6.3). In addition, the metal::precise and metal::fast nested namespaces are also available and provide an explicit way to select the fast or precise variant of these geometric functions.

## 6.9 Synchronization and SIMD-Group Functions

You can use synchronization and SIMD-group functions in:

- [[kernel]] functions

- [[fragment]] functions

- [[visible]] functions that kernel or fragment functions call

## 6.9.1 Threadgroup and SIMD-Group Synchronization Functions

The <metal_compute> header defines the synchronization functions in Table 6.12, which lists threadgroup and SIMD-group synchronization functions it supports.

<div align="center">

Table 6.12. Synchronization compute function in the Metal standard library

</div>

<table border="1"><tr><td>Built-in threadgroup function</td><td>Description</td></tr><tr><td>void threadgroup_barrier(mem_flags flags)</td><td>All threads in a threadgroup executing the kernel, fragment, mesh, or object need to execute this function before any thread can continue execution beyond the threadgroup_barrier.</td></tr><tr><td>void simdgroup_barrier(mem_flags flags)
macOS: Metal 2 and later
iOS: Metal 1.2 and later</td><td>All threads in a SIMD-group executing the kernel, fragment, mesh, or object need to execute this function before any thread can continue execution beyond the simdgroup_barrier.</td></tr></table>

A barrier function (threadgroup_barrier or simdgroup_barrier) acts as an execution and memory barrier. All threads in a threadgroup (or SIMD-group) executing the kernel need to encounter the threadgroup_barrier (or simdgroup_barrier) function. On Apple silicon, a thread that has ended no longer participates or blocks remaining threads at a barrier.

If threadgroup_barrier (or simdgroup_barrier) is inside a conditional statement and if any thread enters the conditional statement and executes the barrier function, then all threads in the threadgroup (or SIMD-group) need to enter the conditional and execute the barrier function.

If threadgroup_barrier (or simdgroup_barrier) is inside a loop, for each iteration of the loop, if any thread in the threadgroup (or SIMD-group) executes the barrier, then all threads

in the threadgroup (or SIMD-group) need to execute the barrier function before any threads continue execution beyond the barrier function.

The threadgroup_barrier (or simdgroup_barrier) function can also queue a memory fence (for reads and writes) to ensure the correct ordering of memory operations to threadgroup or device memory.

Table 6.13 describes the bit field values for the mem_flags argument to threadgroup_barrier and simdgroup_barrier. The mem_flags argument ensures the correct memory is in the correct order between threads in the threadgroup or SIMD-group (for threadgroup_barrier or simdgroup_barrier), respectively.

<div align="center">

Table 6.13. Memory flag enumeration values for barrier functions

</div>

<table border="1"><tr><td>Memory flags (mem_flags)</td><td>Description</td></tr><tr><td>mem_none</td><td>The flag sets threadgroup_barrier or simdgroup_barrier to only act as an execution barrier and doesn&#x27;t apply a Memory fence.</td></tr><tr><td>mem_device</td><td>The flag ensures the GPU correctly orders the memory operations to device memory for threads in the threadgroup or SIMD-group.</td></tr><tr><td>mem_threadgroup</td><td>The flag ensures the GPU correctly orders the memory operations to threadgroup memory for threads in a threadgroup or SIMD-group.</td></tr><tr><td>mem_texture macOS: Metal 1.2 and later iOS: Metal 2 and later</td><td>The flag ensures the GPU correctly orders the memory operations to texture memory for threads in a threadgroup or SIMD-group for a texture with the read_write access qualifier.</td></tr><tr><td>mem_threadgroup_imageblock</td><td>The flag ensures the GPU correctly orders the memory operations to threadgroup imageblock memory for threads in a threadgroup or SIMD-group.</td></tr><tr><td>mem_object_data</td><td>The flag ensures the GPU correctly orders the memory operations to object_data memory for threads in the threadgroup or SIMD-group.</td></tr></table>

## 6.9.2 SIMD-Group Functions

The <metal_simdgroup> header defines the SIMD-group functions for kernel and fragment functions. macOS supports SIMD-group functions in Metal 2 and later, and iOS supports most SIMD-group functions in Metal 2.2 and later. Table 6.14 and Table 6.15 list the SIMD-group functions and their availabilities in iOS and macOS. See the Metal Feature Set Tables to determine which GPUs support each table.

SIMD-group functions allow threads in a SIMD-group (see section 4.4.1) to share data without using threadgroup memory or requiring any synchronization operations, such as a barrier.

An active thread is a thread that is executing. An inactive thread is a thread that is not executing. For example, a thread may not be active due to flow control or when a task has insufficient work to fill the group. A thread needs to only read data from another active thread in the SIMD-group.

Helper threads may also be active and inactive. For example, if a helper thread finishes executing, it becomes an inactive helper thread. Helper threads for SIMD-group functions can be active or inactive. Use simd_is_helper_thread() (see Table 6.14) to inspect whether a thread is a helper thread.

Table 6.14 uses the placeholder T to represent a scalar or vector of any integer or floating-point type, except:

- bool

- long

- ulong

- void

- size_t

- ptrdiff_t

For bitwise operations, Ti needs to be an integer scalar or vector.

See 6.9.2.1 after the table for examples that use SIMD-group functions.

<div align="center">

Table 6.14. SIMD-Group permute functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in SIMD-group functions</td><td>Description</td></tr><tr><td>simd_vote simd_active_threads_mask()
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Returns a simd_vote mask that represents the active threads.
This function is equivalent to simd_ballot(true)and sets the bits that represent active threads to1,and inactive Threads to0.</td></tr><tr><td>bool simd_all(bool expr)
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Returns true if all active threads evaluate expr to true.</td></tr><tr><td>bool simd_any(bool expr)
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Returns true if at least one active thread evaluates Expr to true.</td></tr><tr><td>simd_vote simd_ballot(bool expr)
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Returns a wrapper type—see the simd_vote example—around a bitmask of the evaluation of the Boolean expression for all active threads in the SIMD-group for which expr is true.The function sets the bits that correspond to inactive threads to0.</td></tr></table>

<table border="1"><tr><td>Built-in SIMD-group functions</td><td>Description</td></tr><tr><td>T simd_broadcast(T data,ushort broadcast_lane_id)
macOS: Metal 2 and later
iOS: Metal 2.2 and later</td><td>Broadcasts data from the thread whose SIMD lane ID is equal to broadcast_lane_id.
The specification doesn&#x27;t define the behavior when broadcast_lane_id isn&#x27;t a valid SIMD lane ID or isn&#x27;t the same for all threads in a SIMD-group.</td></tr><tr><td>T simd_broadcast_first(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Broadcasts data from the first active thread—the active thread with the smallest index—in the SIMD-group to all active threads.</td></tr><tr><td>bool simd_is_first()
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Returns true if the current thread is the first active thread—the active thread with the smallest index—in the current SIMD-group; otherwise, false.</td></tr><tr><td>T simd_shuffle(T data,ushort simd_lane_id)
macOS: Metal 2 and later
iOS: Metal 2.2 and later</td><td>Returns data from the thread whose SIMD lane ID is simd_lane_id.The simd_lane_id needs to be a valid SIMD lane ID but doesn&#x27;t have to be the same for all threads in the SIMD-group.</td></tr><tr><td>T simd_shuffle_and_fill_down(T data,T filling_data,ushort delta)
All OS: Metal 2.4 and later</td><td>Returns data or filling_data from the thread whose SIMD lane ID is the sum of the caller&#x27;s SIMD lane ID and delta.
If the sum is greater than the SIMD-group size,the function copies values from the lower delta lanes of filling_data into the upper delta lanes of data.
The value for delta needs to be the same for all threads in a SIMD-group.</td></tr><tr><td>T simd_shuffle_and_fill_down(T data,T filling_data,ushort modulo)
All OS: Metal 2.4 and later</td><td>Returns data or filling_data for each vector from the thread whose SIMD lane ID is the sum of the caller&#x27;s SIMD lane ID and delta.
If the sum is greater than modulo,the function copies values from the lower delta lanes of filling_data into the upper delta lanes of data.
The value of delta needs to be the same for all threads in a SIMD-group.
The modulo parameter defines the vector width that splits the SIMD-group into</td></tr></table>

<table border="1"><tr><td>Built-in SIMD-group functions</td><td>Description</td></tr><tr><td></td><td>separate vectors and must be2,4,8,16,or32.</td></tr><tr><td>T simd_shuffle_and_fill_up(T data,T filling_data,ushort delta)
All OS:Metal2.4and later</td><td>Returns data or filling_data from the thread whose SIMD lane ID is the difference from the caller&#x27;s SIMD lane ID minusdelta.
If the difference is negative,the operation copies values from the upperdelta lanes of filling_data to the lowerdelta lanes of data.
The value ofdelta needs to be the same for all threads in a SIMD-group.</td></tr><tr><td>T simd_shuffle_and_fill_up(T data,T filling_data,ushort modulo)
All OS:Metal2.4and later</td><td>Returns data or filling_data for each vector from the thread whose SIMD lane ID is the difference from the caller&#x27;s SIMD lane ID minusdelta.
If the difference is negative,the operation copies values from the upperdelta lanes of filling_data to the lowerdelta lanes of data.
The value ofdelta needs to be the same for all threads in a SIMD-group.
The modulo parameter defines the vector width that splits the SIMD-group into separate vectors and must be2,4,8,16,or32.</td></tr><tr><td>T simd_shuffle_down(T data,ushort delta)
macOS:Metal2and later
iOS:Metal2.2and later</td><td>Returns data from the thread whose SIMD lane ID is the sum of caller&#x27;s SIMD lane ID anddelta.
The value fordelta needs to be the same for all threads in the SIMD-group.
This function doesn&#x27;t modify the upperdelta lanes of data because it doesn&#x27;t wrap values around the SIMD-group.</td></tr><tr><td>T simd_shuffle_rotate_down(T data,ushort delta)
macOS:Metal2.1and later
iOS:Metal2.2and later</td><td>Returns data from the thread whose SIMD lane ID is the sum of caller&#x27;s SIMD lane ID anddelta.
The value fordelta needs to be the same for all threads in the SIMD-group.
This function wraps values around the SIMD-group.</td></tr></table>

<table border="1"><tr><td>Built-in SIMD-group functions</td><td>Description</td></tr><tr><td>T simd_shuffle_rotate_up(T data,ushort delta)
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Returns data from the thread whose SIMD lane ID is the difference from the caller&#x27;s SIMD lane ID minus delta.
The value of delta needs to be the same for all threads in a SIMD-group.
This function wraps values around the SIMD-group.</td></tr><tr><td>T simd_shuffle_up(T data,ushort delta)
macOS: Metal 2 and later
iOS: Metal 2.2 and later</td><td>Returns data from the thread whose SIMD lane ID is the difference from the caller&#x27;s SIMD lane ID minus delta.
The value of delta needs to be the same for all threads in a SIMD-group.
This function doesn&#x27;t modify the lower delta lanes of data because it doesn&#x27;t wrap values around the SIMD-group.</td></tr><tr><td>Ti simd_shuffle_xor(Ti value,ushort mask)
macOS: Metal 2 and later
iOS: Metal 2.2 and later</td><td>Returns data from the thread whose SIMD lane ID is equal to the bitwise XOR(^) of the caller&#x27;s SIMD lane ID and mask.The value of mask needs to be the same for all threads in a SIMD-group.</td></tr></table>

<div align="center">

Table 6.15. SIMD-Group reduction functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in SIMD-group functions</td><td>Description</td></tr><tr><td>Ti simd_and(Ti data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the bitwise AND (&amp;) of data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.</td></tr><tr><td>bool simd_is_helper_thread()
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns true if the current thread is a helper thread; otherwise, false.
You call this function from a fragment function or another function that your fragment function calls; otherwise, it may trigger a compile-time error.</td></tr><tr><td>T simd_max(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns data with the highest value from across all active threads in the SIMD-group and broadcasts that value to all active threads in the SIMD-group.</td></tr></table>

<table border="1"><tr><td>Built-in SIMD-group functions</td><td>Description</td></tr><tr><td>T simd_min(T data)
macOS: Metal 2.1 and later.
iOS: Metal 2.3 and later.</td><td>Returns data with the lowest value from across all active threads in the SIMD-group and broadcasts that value to all active threads in the SIMD-group.</td></tr><tr><td>Ti simd_or(Ti data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the bitwise OR(|) of data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.</td></tr><tr><td>T simd_prefix_exclusive_product(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>For a given thread, returns the product of the input values in data for all active threads with a lower index in the SIMD-group. The first thread in the group, returns T(1).</td></tr><tr><td>T simd_prefix_exclusive_sum(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>For a given thread, returns the sum of the input values in data for all active threads with a lower index in the SIMD-group. The first thread in the group, returns T(0).</td></tr><tr><td>T simd_prefix_inclusive_product(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>For a given thread, returns the product of the input values in data for all active threads with a lower or the same index in the SIMD-group.</td></tr><tr><td>T simd_prefix_inclusive_sum(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>For a given thread, returns the sum of the input values in data for all active threads with a lower or the same index in the SIMD-group.</td></tr><tr><td>T simd_product(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the product of the input values in data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.</td></tr><tr><td>T simd_sum(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the sum of the input values in data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.</td></tr></table>

<table border="1"><tr><td>Built-in SIMD-group functions</td><td>Description</td></tr><tr><td>Ti simd_xor(Ti data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the bitwise XOR(^) of data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.</td></tr></table>

## 6.9.2.1 Examples

To demonstrate the shuffle functions, start with this SIMD-group's initial state:

<table border="1"><tr><td>SIMD Lane ID</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td></tr><tr><td>data</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td><td>i</td><td>j</td><td>K</td><td>l</td><td>m</td><td>n</td><td>o</td><td>p</td></tr></table>

The simd_shuffle_up() function shifts each SIMD-group upward by delta threads. For example, with a delta value of 2, the function:

- Shifts the SIMD lane IDs down by two

- Marks the lower two lanes as invalid

<table border="1"><tr><td>Computed SIMD lane ID</td><td>-2</td><td>-1</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td></tr><tr><td>valid</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>data</td><td>a</td><td>b</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td><td>i</td><td>j</td><td>k</td><td>l</td><td>m</td><td>n</td></tr></table>

The simd_shuffle_up() function is a no-wrapping operation that doesn't affect the lower delta lanes.

Similarly, the simd_shuffle_down() function shifts each SIMD-group downward by the delta threads. Starting with the same initial SIMD-group state, with a delta value of 2, the function:

- Shifts the SIMD lane IDs up by two

- Marks the upper two lanes as invalid

<table border="1"><tr><td>Computed SIMD lane ID</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td></tr><tr><td>valid</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>data</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td><td>i</td><td>j</td><td>k</td><td>l</td><td>m</td><td>n</td><td>o</td><td>p</td><td>o</td><td>p</td></tr></table>

The simd_shuffle_down() function is a no-wrapping operation that doesn't affect the upper delta lanes.

To demonstrate the shuffle-and-fill functions, start this SIMD-group's initial state:

<table border="1"><tr><td>SIMD lane ID</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td></tr><tr><td>data</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td><td>s</td><td>t</td><td>u</td><td>v</td><td>w</td><td>x</td><td>y</td><td>z</td></tr><tr><td>filling</td><td>fa</td><td>fb</td><td>fc</td><td>fd</td><td>fe</td><td>ff</td><td>fg</td><td>fh</td><td>fs</td><td>ft</td><td>fu</td><td>fv</td><td>fw</td><td>fx</td><td>fy</td><td>fz</td></tr></table>

The simd_shuffle_and_fill_up() function shifts each SIMD-group upward by delta threads — similar to simd_shuffle_up() — and assigns the values from the upper filling lanes to the lower data lanes by wrapping the SIMD lane IDs. For example, with a delta value of 2, the function:

- Shifts the SIMD lane IDs down by two

- Assigns the upper two lanes of filling to the lower two lanes of data

<table border="1"><tr><td>Computed SIMD lane ID</td><td>-2</td><td>-1</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td></tr><tr><td>data</td><td>fy</td><td>fz</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td><td>s</td><td>t</td><td>u</td><td>v</td><td>w</td><td>x</td></tr></table>

The simd_shuffle_and_fill_up() function with the modulo parameter splits the SIMD group into vectors, each with size modulo, and shifts each vector by the delta threads. For example, with a modulo value of 8 and a delta value of 2, the function:

- Shifts the SIMD lane IDs down by two

- Assigns the upper two lanes of each vector in filling to the lower two lanes of each vector in data

<table border="1"><tr><td>Computed SIMD lane ID</td><td>-2</td><td>-1</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>-2</td><td>-1</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>data</td><td>fg</td><td>fh</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>f</td><td>fy</td><td>fz</td><td>s</td><td>t</td><td>u</td><td>v</td><td>w</td><td>x</td></tr></table>

The simd_shuffle_and_fill_down() function shifts each SIMD-group downward by delta threads like simd_shuffle_down() and assigns the values from the lower filling lanes to the upper data lanes by wrapping the SIMD lane IDs. For example, with a delta value of 2, the function:

- Shifts the SIMD lane IDs up by two

- Assigns the lower two lanes of filling to the upper two lanes of data

<table border="1"><tr><td>Computed SIMD lane ID</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td></tr><tr><td>data</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td><td>s</td><td>t</td><td>u</td><td>v</td><td>w</td><td>x</td><td>y</td><td>z</td><td>fa</td><td>fb</td></tr></table>

The simd_shuffle_and_fill_down() function with the modulo parameter splits the SIMD-group into vectors, each with size modulo and shifts each vector by the delta threads. For example, with a modulo value of 8 and a delta value of 2, the function:

- Shifts the SIMD lane IDs up by two

- Assigns the lower two lanes of each vector in filling to the upper two lanes of each vector in data

<table border="1"><tr><td>Computed SIMD lane ID</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td></tr><tr><td>data</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td><td>fa</td><td>fb</td><td>u</td><td>v</td><td>w</td><td>x</td><td>y</td><td>z</td><td>fs</td><td>ft</td></tr></table>

Below is an example of how to use these SIMD functions to perform a reduction operation:

```c

kernel void

reduce(const device int *input [[buffer(0)]],

    device atomic_int *output [[buffer(1)]],

    threadgroup int *ldata [[threadgroup(0)]],

    uint gid [[thread_position_in_grid]],

    uint lid [[thread_position_in_threadgroup]],

    uint lsize [[threads_per_threadgroup]],

    uint simd_size [[threads_per_simdgroup]],

    uint simd_lane_id [[thread_index_in_simdgroup]],

    uint simd_group_id [[simdgroup_index_in_threadgroup]])

{

    // Perform the first level of reduction.

    // Read from device memory, write to threadgroup memory.

    int val = input[gid] + input[gid + lsize];

    for (uint s=lsize/simd_size; s>simd_size; s/=simd_size)

    {

        // Perform per-SIMD partial reduction.

        for (uint offset=simd_size/2; offset>0; offset/=2)

            val += simd_shuffle_down(val, offset);

        // Write per-SIMD partial reduction value to

        // threadgroup memory.

        if (simd_lane_id == 0)

            ldata[simd_group_id] = val;

        // Wait for all partial reductions to complete.

        threadgroup_barrier(mem_flags::mem_threadgroup);

        val = (lid < s) ? ldata[lid] : 0;

    }

    // Perform final per-SIMD partial reduction to calculate

    // the threadgroup partial reduction result.

    for (uint offset=simd_size/2; offset>0; offset/=2)

        val += simd_shuffle_down(val, offset);

    // Atomically update the reduction result.

    if (lid == 0)

        atomic_fetch_add_explicit(output, val,

                                    memory_order_relaxed);

}

```

The simd_active_threads_mask and simd_ballot functions use the simd_vote wrapper type (see below), which can be explicitly cast to its underlying type represented by vote_t.

```cpp

class simd_vote {

public:

    explicit constexpr simd_vote(vote_t v = 0);

    explicit constexpr operator vote_t() const;

    // Returns true if all bits corresponding to threads in the

    // SIMD-group are set.

    // You can use all() with the return value of simd_ballot(expr)

    // to determine if all threads are active.

    bool all() const;

    // Returns true if any bit corresponding to a valid thread in

    // the SIMD-group is set.

    // You can use any() with the return value of simd_ballot(expr)

    // to determine if at least one thread is active.

    bool any() const;

private:

    // bit i in v represents the 'vote' for the thread in the

    // SIMD-group at index i

    uint64_t v;

};

```

Note that simd_all(expr) is different from simd_ballot(expr).all():

- simd_all(expr) returns true if all active threads evaluate expr to true.

- simd_ballot(expr).all() returns true if all threads were active and evaluated the expr to true. (simd_vote::all() does not look at which threads are active.)

The same logic applies to simd_any, simd_vote::any(), and to the equivalent quad functions listed in section 6.9.3.

On hardware with fewer than 64 threads in a SIMD-group, the value of the top bits in simd_vote is undefined. Because you can initialize these bits, do not assume that the top bits are set to 0.

## 6.9.3 Quad-Group Functions

macOS: Metal 2.1 and later support quad-group functions.

iOS: Metal 2 and later support some quad-group functions, including quad_broadcast, quad_shuffle, quad_shuffle_up, quad_shuffle_down, and quad_shuffle_xor.

A quad-group function is a SIMD-group function (see section 6.9.2) with an execution width of 4. The active and inactive thread terminology is the same as in section 6.9.2.

Helper threads only execute to compute gradients for quad-groups in a fragment shader and then become inactive.

Kernels and fragment functions can call the quad-group functions listed in Table 6.17 and Table 6.18. Threads may only read data from another active thread in a quad-group. See the Metal Feature Set Tables to determine which GPUs support each table.

The placeholder T for Table 6.17 and Table 6.18 represents a scalar or vector of any integer or floating-point type, except:

- bool

- void

- size_t

- ptrdiff_t

For bitwise operations, T needs to be an integer scalar or vector.

<div align="center">

Table 6.16. Quad-group function in the Metal standard library

</div>

<table border="1"><tr><td>Built-in quad-group functions</td><td>Description</td></tr><tr><td>quad_vote quad_ballot(bool expr)
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Returns a quad_vote bitmask where each bit indicates where the Boolean expression expr evaluates to true for active threads in the quad-group. The function sets the bits that correspond to inactive threads to 0. See an example at the end of this section.</td></tr></table>

<div align="center">

Table 6.17. Quad-group permute functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in quad-group functions</td><td>Description</td></tr><tr><td>T quad_broadcast(T data,ushort broadcast_lane_id)
macOS: Metal 2 and later
iOS: Metal 2 and later</td><td>Broadcasts data from the thread whose quad lane ID is broadcast_lane_id.The value for broadcast_lane_id needs to be a valid quad lane ID that&#x27;s the same for all threads in a quad-group.</td></tr><tr><td>T quad_broadcast_first(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Broadcasts data from the first active thread—the active thread with the smallest index in the quad-group to all active threads.</td></tr><tr><td>T quad_shuffle(T data,ushort quad_lane_id)
macOS: Metal 2 and later
iOS: Metal 2 and later</td><td>Returns data from the thread whose quad lane ID is quad_lane_id. The value for quad_lane_id needs to be a valid lane ID and may differ across threads in the quad-group.</td></tr><tr><td>T quad_shuffle_and_fill_down(T data,T filling_data,ushort delta)
All OS: Metal 2.4 and later</td><td>Returns data or filling_data from the thread whose quad lane ID is the sum of the caller&#x27;s quad lane ID and delta.If the sum is greater than the quad-group size,the function copies values from the lower delta lanes of filling_data into the upper delta lanes of data.</td></tr></table>

<table border="1"><tr><td>Built-in quad-group functions</td><td>Description</td></tr><tr><td></td><td>The value for delta needs to be the same for all threads in a quad-group.</td></tr><tr><td>T quad_shuffle_and_fill_down(T data,T filling_data,ushort delta,ushort modulo)
All OS:Metal2.4and later</td><td>Returns data or filling_data for each vector,from the thread whose quad lane ID is the sum of caller&#x27;s quad lane ID and delta.If the sum is greater than the quad-group size,the function copies values from the lower delta lanes of filling_data into the upper delta lanes of data.The value of delta needs to be the same for all threads in a quad-group.The modulo parameter defines the vector width that splits the quad-group into separate vectors and must be 2 or 4.</td></tr><tr><td>T quad_shuffle_and_fill_up(T data,T filling_data,ushort delta)
All OS:Metal2.4and later</td><td>Returns data or filling_data from the thread whose quad lane ID is the difference from the caller&#x27;s quad lane ID minus delta.If the difference is negative,the operation copies values from the upper delta lanes of filling_data to the lower delta lanes of data.If the difference is negative,the function shuffles data from filling_data into the lower delta lanes.The value of delta needs to be the same for all threads in a quad-group.</td></tr><tr><td>T quad_shuffle_and_fill_up(T data,T filling_data,ushort delta,ushort modulo)
All OS:Metal2.4and later</td><td>Returns data or filling_data for each vector from the thread whose quad lane ID is the difference from the caller&#x27;s quad lane ID minus delta.If the difference is negative,the operation copies values from the upper delta lanes of filling_data to the lower delta lanes of data.The value of delta needs to be the same for all threads in a quad-group.The modulo parameter defines the width that splits the quad-group into separate vectors and must be 2 or 4.</td></tr></table>

<table border="1"><tr><td>Built-in quad-group functions</td><td>Description</td></tr><tr><td>T quad_shuffle_down(T data,ushort delta)
macOS: Metal 2 and later
iOS: Metal 2 and later</td><td>Returns data from the thread whose quad lane ID is the sum of the caller&#x27;s quad lane ID and delta.
The value for delta needs to be the same for all threads in a quad-group.
The function doesn&#x27;t modify the upper delta lanes of data because it doesn&#x27;t wrap values around the quad-group.</td></tr><tr><td>T quad_shuffle_rotate_down(T data,ushort delta)
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Returns data from the thread whose quad lane ID is the sum of the caller&#x27;s quad lane ID and delta.
The value for delta needs to be the same for all threads in a quad-group.
This function wraps values around the quad-group.</td></tr><tr><td>T quad_shuffle_rotate_up(T data,ushort delta)
macOS: Metal 2.1 and later
iOS: Metal 2.2 and later</td><td>Returns data from the thread whose quad lane ID is the difference from the caller&#x27;s quad lane ID minus delta.
The value for delta needs to be the same for all threads in a quad-group.
This function wraps values around the quad-group.</td></tr><tr><td>T quad_shuffle_up(T data,ushort delta)
macOS: Metal 2 and later
iOS: Metal 2 and later</td><td>Returns data from thread whose quad lane ID is the difference from the caller&#x27;s quad lane ID minus delta.
The value for delta needs to be the same for all threads in a quad-group.
This function doesn&#x27;t modify the lower delta lanes of data because it doesn&#x27;t wrap values around the quad-group.</td></tr><tr><td>T quad_shuffle_xor(T value,ushort mask)
macOS: Metal 2 and later
iOS: Metal 2 and later</td><td>Returns data from the thread whose quad lane ID is a bitwise XOR(^) of the caller&#x27;s quad lane ID and mask. The value of mask needs to be the same for all threads in a quad-group.</td></tr></table>

<div align="center">

Table 6.18. Quad-group reduction functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in quad-group functions</td><td>Description</td></tr><tr><td>quad_vote quad_active_threads_mask()
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns a quad_vote mask that represents the active threads.
The function is equivalent to quad_ballot(true) and sets the bits that represent active threads to1 and inactive threads to0.</td></tr><tr><td>bool quad_all(bool expr)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns true if all active threads evaluate expr to true.</td></tr><tr><td>T quad_and(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the bitwise AND(&amp;) of data across all active threads in the quad-group and broadcasts the result to all active threads in the quad-group.</td></tr><tr><td>bool quad_any(bool expr)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns true if at least one active thread evaluates expr to true.</td></tr><tr><td>bool quad_is_first()
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns true if the current thread is the first active thread—the active thread with the smallest index—in the current quad-group; otherwise,false.</td></tr><tr><td>bool quad_is_helper_thread()
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns true if the current thread is a helper thread; otherwise,false.
You call this function from a fragment function or another function that your fragment function calls; otherwise,it may trigger a compile-time error.</td></tr><tr><td>T quad_max(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns data with the highest value from across all active threads in the quad-group and broadcasts that value to all active threads in the quad-group.</td></tr><tr><td>T quad_min(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns data with the lowest value from across all active threads in the quad-group and broadcasts that value to all active threads in the quad-group.</td></tr></table>

<table border="1"><tr><td>Built-in quad-group functions</td><td>Description</td></tr><tr><td>T quad_or(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the bitwise OR(|) of data across all active threads in the quad-group and broadcasts the result to all active threads in the quad-group.</td></tr><tr><td>T quad_prefix_exclusive_product(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>For a given thread, returns the product of the input values in data for all active threads with a lower index in the quad-group. For the first thread in the group, return T(1).</td></tr><tr><td>T quad_prefix_exclusive_sum(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>For a given thread, returns the sum of the input values in data for all active threads with a lower index in the quad-group. For the first thread in the group, return T(0).</td></tr><tr><td>T quad_prefix_inclusive_product(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>For a given thread, returns the product of the input values in data for all active threads with a lower or the same index in the quad-group.</td></tr><tr><td>T quad_prefix_inclusive_sum(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>For a given thread, returns the sum of the input values in data for all active threads with a lower or the same index in the quad-group.</td></tr><tr><td>T quad_product(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the product of the input values in data across all active threads in the quad-group and broadcasts the result to all active threads in the quad-group.</td></tr><tr><td>T quad_sum(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the sum of the input values in data across all active threads in the quad-group and broadcasts the result to all active threads in the quad-group.</td></tr><tr><td>T quad_xor(T data)
macOS: Metal 2.1 and later
iOS: Metal 2.3 and later</td><td>Returns the bitwise XOR(^) of data across all active threads in the quad-group and broadcasts the result to all active threads in the quad-group.</td></tr></table>

In a kernel function, quads divide across the SIMD-group. In a fragment function, the lane ID represents the fragment location in a 2 x 2 quad:

- Lane ID 0 is the upper-left pixel

- Lane ID 1 is the upper-right pixel

- Lane ID 2 is the lower-left pixel

- Lane ID 3 is the lower-right pixel

To demonstrate the shuffle functions, start with this quad-group's initial state:

<table border="1"><tr><td>Quad lane ID</td><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td>data</td><td>a</td><td>b</td><td>c</td><td>d</td></tr></table>

The quad_shuffle_up() function shifts each quad-group upward by delta threads. For example, with a delta value of 2, the function:

- Shifts the quad lane IDs down by two

- Marks the lower two lanes as invalid

<table border="1"><tr><td>Computed quad lane ID</td><td>-2</td><td>-1</td><td>0</td><td>1</td></tr><tr><td>valid</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>data</td><td>a</td><td>b</td><td>a</td><td>b</td></tr></table>

The quad_shuffle_up() function is a no wrapping operation that doesn't affect the lower delta lanes.

Similarly, quad_shuffle_down() function shifts each quad-group downward by delta threads. Starting with the same initial quad-group state, with a delta of 2, the function:

- Shifts the quad lane IDs up by two

- Marks the upper two lanes as invalid

<table border="1"><tr><td>Computed quad lane ID</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>valid</td><td>1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>data</td><td>c</td><td>d</td><td>c</td><td>d</td></tr></table>

The quad_shuffle_down() function is a no wrapping operation that doesn't affect the upper delta lanes.

To demonstrate the shuffle-and-fill functions, start this quad-group's initial state:

<table border="1"><tr><td>Quad lane ID</td><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td>data</td><td>a</td><td>b</td><td>c</td><td>d</td></tr><tr><td>filling</td><td>fa</td><td>fb</td><td>fc</td><td>fd</td></tr></table>

The quad_shuffle_and_fill_up() function shifts each quad-group upward by the delta threads — similar to quad_shuffle_up() — and assigns the values from the upper filling lanes to the lower data lanes by wrapping the quad lane IDs. For example, with a delta value of 2, the function:

- Shifts the quad lane IDs down by two

- Assigns the upper two lanes of filling to the lower two lanes of data

<table border="1"><tr><td>Computed quad lane ID</td><td>-2</td><td>-1</td><td>0</td><td>1</td></tr><tr><td>data</td><td>fc</td><td>fd</td><td>a</td><td>b</td></tr></table>

The quad_shuffle_and_fill_up() function with the modulo parameter splits the quad-group into vectors, each with size modulo and shifts each vector by the delta threads. For example, with a modulo value of 2 and a delta value of 1, the function:

- Shifts the quad lane IDs down by one

- Assigns the upper lane of each vector in filling to the lower lane of each vector in data

<table border="1"><tr><td>Computed quad lane ID</td><td>-1</td><td>0</td><td>-1</td><td>0</td></tr><tr><td>data</td><td>fb</td><td>a</td><td>fd</td><td>c</td></tr></table>

The quad_shuffle_and_fill_down() function shifts each quad-group downward by delta threads — similar to quad_shuffle_down() — and assigns the values from the lower filling lanes to the upper data lanes by wrapping the quad lane IDs. For example, with a delta value of 2, the function:

- Shifts the quad lane IDs up by two

- Assigns the lower two lanes of filling to the upper two lanes of data

<table border="1"><tr><td>Computed quad lane ID</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>data</td><td>c</td><td>d</td><td>fa</td><td>fb</td></tr></table>

The quad_shuffle_and_fill_down() function with the modulo parameter splits the quad-group into vectors, each with size modulo and shifts each vector by the delta threads. For example, with a modulo value of 2 and a delta value of 1, the function:

- Shifts the quad lane IDs up by one

- Assigns the lower lane of each vector in filling to the upper lane of each vector in data

<table border="1"><tr><td>Computed quad lane ID</td><td>1</td><td>2</td><td>1</td><td>2</td></tr><tr><td>data</td><td>b</td><td>fa</td><td>d</td><td>fc</td></tr></table>

The quad_ballot function uses the quad_vote wrapper type, which can be explicitly cast to its underlying type. (In the following example, note use of vote_t to represent an underlying type.)

```cpp

class quad_vote {

public:

    typedef unsigned vote_t;

    explicit constexpr quad_vote(vote_t v = 0);

    explicit constexpr operator vote_t() const;

    // Returns true if all bits corresponding to threads in the

    // quad-group (the four bottom bits) are set.

    bool all() const;

    // Returns true if any bit corresponding to a thread in the

    // quad-Group is set.

    bool any() const;

};

```

The quad_vote constructor masks out the top bits (that is, other than the four bottom bits). Therefore, Metal clears the upper bits, and the bottom four bits don't change when you cast to vote_t.

## 6.10 Graphics Functions

The graphics functions in this section and its subsections are defined in the header <metal_graphics>. You can only call these graphics functions from a fragment function.

## 6.10.1 Fragment Functions

You can only call the functions in this section (listed in Table 6.19, Table 6.20, and Table 6.21) inside a fragment function (see section 5.1.2) or inside a function called from a fragment function. Otherwise, the behavior is undefined and may result in a compile-time error.

Fragment function helper threads may be created to help evaluate derivatives (explicit or implicit) for use with a fragment thread(s). Fragment function helper threads execute the same code as the other fragment threads, but do not have side effects that modify the render targets or any other memory that can be accessed by the fragment function. In particular:

- Fragments corresponding to helper threads are discarded when the fragment function execution is complete without any updates to the render targets.

- Stores and atomic operations to buffers and textures performed by helper threads have no effect on the underlying memory associated with the buffer or texture.

## 6.10.1.1 Fragment Functions - Derivatives

Metal includes the functions in Table 6.19 to compute derivatives. T is one of float, float2, float3, float4, half, half2, half3, or half4.

Derivatives are undefined within nonuniform control flow.

Note: In Metal 2.2 and earlier, discard_fragment could make the control flow nonuniform. In Metal 2.3 and later, discard_fragment does not affect whether the control flow is considered nonuniform or not. See Section 6.10.1.3 for more information.

<div align="center">

Table 6.19. Derivatives fragment functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in fragment functions</td><td>Description</td></tr><tr><td>T dfdx(T p)</td><td>Returns a high precision partial derivative of the specified value with respect to the screen space x coordinate.</td></tr><tr><td>T dfdy(T p)</td><td>Returns a high precision partial derivative of the specified value with respect to the screen space y coordinate.</td></tr><tr><td>T fwidth(T p)</td><td>Returns the sum of the absolute derivatives in x and y using local differencing for p; that is, fabs(dfdx(p)) + fabs(dfdy(p))</td></tr></table>

## 6.10.1.2 Fragment Functions - Samples

Metal includes the per-sample functions listed in Table 6.20. get_num_samples and get_sample_position return the number of samples for the color attachment and the sample offsets for a given sample index. For example, for transparency super-sampling, these functions can be used to shade per-fragment but do the alpha test per-sample.

<div align="center">

Table 6.20. Samples fragment functions in the Metal standard library

</div>

<table border="1"><tr><td>Built-in fragment functions</td><td>Description</td></tr><tr><td>uint get_num_samples()</td><td>Returns the number of samples for the multisampled color attachment.</td></tr><tr><td>float2 get_sample_position(uint index)</td><td>Returns the normalized sample offset(x,y) for a given sample index index. Values of x and y are in[0.0...1.0].</td></tr></table>

If you have customized sample positions (set with the setSamplePositions:count method of MTLRenderPassDescriptor), get_sample_position(index) returns the position programmed for the specified index.

## 6.10.1.3 Fragment Functions — Flow Control

The Metal function in Table 6.21 terminates a fragment.

<div align="center">

Table 6.21. Fragment flow control function in the Metal standard library

</div>

<table border="1"><tr><td>Built-in fragment functions</td><td>Description</td></tr><tr><td>void discard_fragment(void)</td><td>Marks the current fragment as terminated and discards this fragment&#x27;s output of the fragment function.</td></tr></table>

Writes to a buffer or texture from a fragment thread made before calling discard_fragment are not discarded.

Multiple fragment threads or helper threads associated with a fragment thread execute together to compute derivatives. In Metal 2.2 and earlier, if any (but not all) of these threads executes the discard_fragment function, the thread is terminated and the behavior of any derivative computations (explicit or implicit) is undefined. In Metal 2.3 and later, discard_fragment marks the fragment as terminated while continuing to execute in parallel and has no effect on whether derivatives are defined. Even though execution continues, the write behavior remains the same as before. The fragment will discard the fragment output and discard all writes to buffer or texture after discard_fragment.

## 6.11 Pull-Model Interpolation

All OS: Metal 2.3 and later support pull-model interpolation.

The interpolant type interpolant<T,P> (section 2.18) and associated methods are defined in <metal_interpolate>. In a fragment function, you explicitly interpolate the values of a interpolant<T,P> type by invoking its methods, as shown below. The interpolant may be sampled and interpolated multiple times, in different modes, and may be passed to other functions to be sampled and interpolated there. Perspective correctness is fixed across all interpolations of the argument by the value of P in its type.

<div align="center">

Table 6.22. Pull-Model interpolant methods

</div>

<table border="1"><tr><td>Interpolant method</td><td>Description</td></tr><tr><td>T interpolate_at_center()</td><td>Sample shader input at the center of a pixel, returning the same value as if the input had type T with [[center_perspective]] or [[center_no_perspective]].</td></tr><tr><td>T interpolate_at_centroid()</td><td>Sample shader input within the covered area of the pixel, returning the same value as if the input had type T with [[centroid_perspective]] or [[centroid_no_perspective]].</td></tr></table>

<table border="1"><tr><td>T interpolate_at_offset(float2 offset)</td><td>Sample shader input at a specified window-coordinate offset from a pixel&#x27;s top-left corner. Allowable offset components are in the range[0.0,1.0) along a 1/16 pixel grid.</td></tr><tr><td>T interpolate_at_sample(uint sample)</td><td>Sample shader input at the location of the specified sample index, returning the same value as if the input had typeT with[[sample_perspective]]or[[sample_no_perspective]]and was in the specified per-sample evaluation of the shader. If a sample of the given index does not exist,the position of interpolation is undefined.</td></tr></table>

## 6.12 Texture Functions

The texture member functions, defined in the header <metal_texture>, listed in this section and its subsections for different texture types include:

- sample - sample from a texture

- sample_compare - sample compare from a texture

- gather - gather from a texture

- gather_compare — gather compare from a texture

- read - sampler-less read from a texture

- write - write to a texture

- texture query (such as get_width, get_height, get_num_mip_levels, get_array_size)

- texture fence

- In Metal 3.1 and later, new atomic texture member functions are supported on 1D texture, 1D texture array, 2D texture, 2D texture array, 3D texture, and texture buffer for int, uint, and ulong color types:

- atomic_load - atomic load from a texture

- atomic_store - atomic store to a texture

- atomic_exchange — atomic exchange a value for a texture

- atomic_compare_exchange_weak — atomic compare and exchange in a texture

- atomic_fetch_op_explicit — atomic fetch and modify where op can be add, and, max, min, or, sub, or xor for int and uint color type

- atomic_max — atomic max in a texture for ulong color type

- atomic_min - atomic min in a texture for ulong color type

Metal 4 adds support for the atomic texture functions for cube texture and cube texture array.

See the Metal Feature Set Tables to determine which GPUs support texture atomics.

Metal 3.2 introduces coherence (see section 2.9).

The texture sample, sample_compare, gather, and gather_compare functions take an offset argument for a 2D texture, 2D texture array, and 3D texture. The offset is an integer value applied to the texture coordinate before looking up each pixel. This integer value can be in the range -8 to +7; the default value is 0.

The texture sample, sample_compare, gather, and gather_compare functions require that you declare the texture with the sample access attribute. The texture read functions require that you declare the texture with the sample, read, or read_write access attribute. The texture write functions require that you declare the texture with the write or read_write access attribute. (For more about access attributes, see section 2.9.)

The texture sample_compare and gather_compare functions are only available for depth texture types.

compare_func sets the comparison test for the sample_compare and gather_compare functions. For more about compare_func, see section 2.10.

Overloaded variants of the texture sample and sample_compare functions with an lod_options argument are available for a 2D texture, 2D texture array, 2D depth texture, 2D depth texture array, 3D texture, cube texture, cube texture array, cube depth texture, and cube depth texture array. (LOD/lod is short for level-of-detail.) The values for lod_options are:

- level(float lod) — Sample from the specified mipmap level.

- bias(float value) - Apply the specified bias to a mipmap level before sampling.

- gradient $ \ast $ (T dPdx, T dPdy) — Apply the specified gradients with respect to the x and y directions. The texture type changes the name and the arguments; for example, for 3D textures, the name is gradient3d and the arguments are float3 type.

- min_lod_clamp(float lod) - Specify lowest mipmap level for sampler access, which restricts sampler access to a range of mipmap levels. (All OS: Support since Metal 2.2.)

In macOS, Metal 2.2 and earlier don't support sample_compare, bias and gradient* functions, and lod needs to be a zero constant. Metal 2.3 and later lift this restriction for Apple silicon.

In Metal 2.2 and later, you can specify a LOD range for a sampler. You can either specify a minimum and maximum mipmap level or use min_lod_clamp to specify just the minimum mipmap level of an open range. When the sampler determines which mipmaps to sample, the selection is clamped to the specified range.

Clamping the LOD is useful where some of the texture data is not available all the time (for example, texture streaming). You can create a texture with all the necessary mipmaps and then can stream image data starting from the smallest mipmaps. When the GPU samples the texture, it clamps to the mipmaps that already have valid data. When you copy larger mipmaps into the texture, you reduce the minimum LOD level. As new data becomes ready, you can change the LOD clamp, which changes the sampling resolution.

The texture sample and sample_compare functions that don't take an explicit LOD or gradients when you don't call them in a fragment function, have a default LOD of 0. In a fragment function, the texture sample and sample_compare functions that don't take an explicit LOD or gradients calculate an implicit LOD by taking the derivative of the texture

coordinate passed to the function. The gather and gather_compare functions you don't call in a fragment function also have a default LOD of 0.

For the gather and gather_compare functions, place the four samples that contribute to filtering into xyzw components in counter-clockwise order, starting with the sample to the lower-left of the queried location. This is the same as nearest sampling with unnormalized texture coordinate deltas at the following locations: （-，+），（+，+），（+，-），（-，-），where the magnitude of the deltas is always half a pixel.

A read from or write to a texture is out-of-bounds if and only if any of these conditions is met:

- the coordinates accessed are out-of-bounds

- the level of detail argument is out-of-bounds

- the texture is a texture array (texture?d_array type), and the array slice argument is out-of-bounds

- the texture is a texturecube or texturecube_array type, and the face argument is out-of-bounds

- the texture is a multisampled texture, and the sample argument is out-of-bounds

For all texture types, an out-of-bounds write to a texture is ignored.

For all texture types:

- For components specified in a pixel format, an out-of-bounds read returns a color with components with the value zero.

- For components unspecified in a pixel format, an out-of-bounds read returns the default value.

- For unspecified color components in a pixel format, the default values are:

- 0, for components other than alpha.

- 1, for the alpha component.

In a pixel format with integer components, the alpha default value is represented as the integral value 0x1. For a pixel format with floating-point or normalized components, the alpha default value is represented as the floating-point value 1.0.

For example, for a texture with the MTLPixelFormatR8Uint pixel format, the default values for unspecified integer components are G = 0, B = 0, and A = 1. For a texture with the MTLPixelFormatR8Unorm pixel format, the default values for unspecified normalized components are G = 0.0, B = 0.0, and A = 1.0. For a texture with depth or stencil pixel format (such as MTLPixelFormatDepth24Unorm_Stencil8 or MTLPixelFormatStencil8), the default value for an unspecified component is undefined.

In macOS, for Metal 2.2 and earlier, lod needs to be 0 for texture write functions. Metal 2.3 and later lift this restriction for Apple silicon.

The following texture member functions are available to support sparse textures:

macOS: Metal 2.3 and later support sparse texture functions. iOS: Metal 2.2 and later support sparse texture functions.

- sparse_sample — sample from a sparse texture

- sparse_sample_compare — sample compare from a sparse texture

- sparse_gather — gather from a sparse texture

- sparse_gather_compare — gather compare from a sparse texture

These sparse texture member functions return a sparse_color structure that contains one or more color values and a residency flag. If any of the accessed pixels is not mapped, resident is set to false.

```cpp

template <typename T>

struct sparse_color {

public:

    constexpr sparse_color(T value, bool resident) thread;

    // Indicates whether all memory addressed to retrieve

    // the value was mapped.

    constexpr bool resident() const thread;

    // Retrieve the color value.

    constexpr T const value() const thread;

};

```

For a sparse texture, to specify the minimum LOD level that the sampler can access, use min_lod_clamp.

## Note:

For sections 6.12.1 through 6.12.16, the following abbreviations are used for the data types of function arguments and return values:

Tv denotes a 4-component vector type based on the templated type <T> for declaring the texture type:

- If T is float, Tv is float4.

- If T is half, Tv is half4.

- If T is int, Tv is int4.

- If T is uint, Tv is uint4.

- If T is short, Tv is short4.

- If T is ushort, Tv is ushort4.

- If T is ulong, Tv is ulong4 (since Metal 3.1)

Metal does not support sampling of textures when T is ulong. Note that not all operations are supported on all types.

In Metal 3.1 and later, texture support atomic functions for element T where T is int, uint, or ulong:

- When the element T is int or uint, the texture on the Metal needs to be either MTLPixelFormatR32Uint or MTLPixelFormatR32Sint.

- When the element T is ulong, the texture on the Metal needs to be MTLPixelFormatRG32Uint.

The semantics of the atomic texture functions are the same as the atomic functions defined in Sec 6.15.

sparse_color<Tv> denotes a sparse_color structure that contains a four-component vector of color values, based on the templated type <T>, and a residency flag. These represent the return values of many sparse texture member functions.

sparse_color<T> denotes a sparse_color structure that contains a single value, based on the templated type <T>, and a residency flag. T typically represents a depth value that a sparse texture member function returns.

The following functions can be used to return the LOD (mip level) computation result for a simulated texture fetch:

macOS: Metal 2.2 and later support sparse texture functions. iOS: Metal 2.3 and later support sparse texture functions.

calculate_unclamped_lod —Calculates the level of detail that would be sampled for the given coordinates, ignoring any sampler parameter. The fractional part of this value contains the mip level blending weights, even if the sampler indicates a nearest mip selection.

calculate_clamped_lod - Similar to calculate_unclamped_lod, but additionally clamps the LOD to stay:

- within the texture mip count limits

- within the sampler's lod_clamp min and max values

- less than or equal to the sampler's max_anisotropy value

Only call the calculate_unclamped_lod and calculate_clamped_lod functions from a fragment function or a function you call with a fragment function; otherwise, the behavior is undefined.

## 6.12.1 1D Texture

This member function samples from a 1D texture.

Tv sample(sampler s, float coord) const

These member functions perform sampler-less reads from a 1D texture. Because mipmaps are not supported for 1D textures, lod needs to be 0:

Tv read(uint coord, uint lod = 0) const

Tv read(ushort coord,

ushort lod = 0) const // All OS: Metal 1.2 and later.

These member functions can write to a 1D texture. Because mipmaps are not supported for 1D textures, lod needs to be 0:

void write(Tv color, uint coord, uint lod = 0)

void write(Tv color, ushort coord,

ushort lod = 0) // All OS: Metal 1.2 and later.

These member functions query a 1D texture. Since mipmaps are not supported for 1D textures, get_num_mip_levels() always return 0, and lod needs to be 0 for get_width():

```cpp

uint get_width(uint lod = 0) const

uint get_num_mip_levels() const

```

sparse_color<Tv> sparse_sample(sampler s, float coord) const

This member function samples from a sparse 1D texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

These member functions perform a sampler-less read from a sparse 1D texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS. Because mipmaps are not supported for 1D textures, lod needs to be 0:

sparse_color<Tv> sparse_read(ushort coord, ushort lod = 0) const

sparse_color<Tv> sparse_read(uint coord, uint lod = 0) const

These member functions execute an atomic load from a 1D texture in Metal 3.1 and later:

Tv atomic_load(uint coord) const

Tv atomic_load(ushort coord) const

These member functions execute an atomic store to a 1D texture in Metal 3.1 and later:

void atomic_store(Tv color, uint coord) const

void atomic_store (Tv color, ushort coord) const

These member functions execute an atomic compare and exchange to a 1D texture in Metal 3.1 and later:

bool atomic_compare_exchange_weak(uint coord, thread Tv *expected,

Tv desired) const

bool atomic_compare_exchange_weak(ushort coord, thread Tv *expected,

Tv desired) const

These member functions execute an atomic exchange to a 1D texture in Metal 3.1 and later:

Tv atomic_exchange(uint coord, Tv desired) const

Tv atomic_exchange(ushort coord, Tv desired) const

These member functions execute an atomic fetch and modify to a 1D texture in Metal 3.1 and later, where op is add, and, max, min, or, sub, or xor for int, and uint color type:

Tv atomic_fetch_op(uint coord, Tv operand)

Tv atomic_fetch_op(ushort coord, Tv operand) const

These member functions execute an atomic min or max to a 1D texture in Metal 3.1 and later:

void atomic_min(uint coord, ulong4 operand)

void atomic_min(ushort coord, ulong4 operand)

void atomic_max(uint coord, ulong4 operand)

void atomic_max(ushort coord, ulong4 operand)

## 6.12.2 1D Texture Array

This member function samples from a 1D texture array:

Tv sample(sampler s, float coord, uint array) const

These member functions perform sampler-less reads from a 1D texture array. Because mipmaps are not supported for 1D textures, lod must be a zero constant:

Tv read(uint coord, uint array, uint lod = 0) const

Tv read(ushort coord, ushort array,

ushort lod = 0) const // All OS: Metal 1.2 and later.

These member functions write to a 1D texture array. Because mipmaps are not supported for 1D textures, lod must be a zero constant:

void write(Tv color, uint coord, uint array, uint lod = 0)

void write(Tv color, ushort coord, ushort array,

ushort lod = 0) // All OS: Metal 1.2 and later.

These member functions query a 1D texture array. Because mipmaps are not supported for 1D textures, get_num_mip_levels() always return 0, and lod must be a zero constant for get_width():

uint get_width(uint lod = 0) const

uint get_array_size() const

uint get_num_mip_levels() const

sparse_color<Tv> sparse_sample(sampler s, float coord, uint array) const

This function samples from a sparse 1D texture array in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

These functions perform a sampler-less read from a sparse 1D texture array in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS. Because mipmaps are not supported for 1D texture arrays, lod must be a zero constant.

sparse_color<Tv> sparse_read(ushort coord, ushort array,

ushort lod = 0) const

sparse_color<Tv> sparse_read(uint coord, uint array,

uint lod = 0) const

These member functions execute an atomic load from a 1D texture array in Metal 3.1 and later:

Tv atomic_load(uint coord, uint array) const

Tv atomic_load(ushort coord, ushort array) const

These member functions execute an atomic store to a 1D texture array in Metal 3.1 and later:

void atomic_store(Tv color, uint coord, uint array) const

void atomic_store (Tv color, ushort coord, ushort array) const

These member functions execute an atomic compare and exchange to a 1D texture array in Metal 3.1 and later:

bool atomic_compare_exchange_weak(uint coord, uint array,

thread Tv *expected,

Tv desired) const

bool atomic_compare_exchange_weak(ushort coord, ushort array,

thread Tv *expected,

Tv desired) const

These member functions execute an atomic exchange to a 1D texture array in Metal 3.1 and later:

Tv atomic_exchange(uint coord, uint array, Tv desired) const

Tv atomic_exchange(ushort coord, ushort array, Tv desired) const

These member functions execute an atomic fetch and modify to a 1D texture array in Metal 3.1 and later, where op is add, and, max, min, or, sub, or xor:

Tv atomic_fetch_op(uint coord, uint array,Tv operand)

Tv atomic_fetch_op(ushort coord, ushort array,Tv operand) const

These member functions execute an atomic min or max to a 1D texture array in Metal 3.1 and later:

void atomic_min(uint coord, uint array, ulong4 operand)

void atomic_min(ushort coord, ushort array, ulong4 operand)

void atomic_max(uint coord, uint array, ulong4 operand)

void atomic_max(ushort coord, ushort array, ulong4 operand)

## 6.12.3 2D Texture

For the functions in this section, the following data types and corresponding constructor functions can specify sampling options (lod_options):

bias(float value)

level(float lod)

gradient2d(float2 dPdx, float2 dPdy)

min_lod_clamp(float lod) // All OS: Metal 2.2 and later.

These member functions sample from a 2D texture:

Tv sample(sampler s, float2 coord, int2 offset = int2(0)) const

Tv sample(sampler s, float2 coord, lod_options options,

    int2 offset = int2(0)) const

Tv sample(sampler s, float2 coord, bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

Tv sample(sampler s, float2 coord, gradient2d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

These member functions perform sampler-less reads from a 2D texture:

Tv read(uint2 coord, uint lod = 0) const

Tv read(ushort2 coord,

    ushort lod = 0) const // All OS: Metal 1.2 and later.

These member functions write to a 2D texture. In macOS, for Metal 2.2 and earlier, lod must be a zero constant. Metal 2.3 and later lift this restriction for Apple silicon.

void write(Tv color, uint2 coord, uint lod = 0)

void write(Tv color, ushort2 coord,

    ushort lod = 0) // All OS: Metal 1.2 and later.

This member functions gathers four samples for bilinear interpolation when sampling a 2D texture:

enum class component {x, y, z, w};

Tv gather(sampler s, float2 coord, int2 offset = int2(0),

    component c = component::x) const

These member functions query a 2D texture query:

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_num_mip_levels() const

These member functions sample from a sparse 2D texture in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_sample(sampler s, float2 coord,

    int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord, bias options,

    int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord,

    level options,

    int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord,

    bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord,

    gradient2d grad_options,

    int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord,

    gradient2d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

These member functions perform a sampler-less read from a sparse 2D texture in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_read(ushort2 coord, ushort lod = 0) const

sparse_color<Tv> sparse_read(uint2 coord, uint lod = 0) const

This member function gathers four samples for bilinear interpolation from a sparse 2D texture in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather(sampler s, float2 coord,

int2 offset = int2(0),

component c = component::x) const

These member functions simulate a texture fetch and return the LOD (mip level) computation result in Metal 2.3 and later in iOS, and in Metal 2.2 and later in macOS:

float calculate_clamped_lod(sampler s, float2 coord);

float calculate_unclamped_lod(sampler s, float2 coord);

Tv atomic_load(uint2 coord) const

Tv atomic_load(ushort2 coord) const

These member functions execute an atomic store to a 2D texture in Metal 3.1 and later:

void atomic_store(Tv color, uint2 coord) const

void atomic_store (Tv color, ushort2 coord) const

These member functions execute an atomic compare and exchange to a 2D texture in Metal 3.1 and later:

bool atomic_compare_exchange_weak(uint2 coord, thread Tv *expected,

Tv desired) const

bool atomic_compare_exchange_weak(ushort2 coord, thread Tv *expected,

Tv desired) const

These member functions execute an atomic exchange to a 2D texture in Metal 3.1 and later:

Tv atomic_exchange(uint2 coord, Tv desired) const

Tv atomic_exchange(ushort2 coord, Tv desired) const

These member functions execute an atomic fetch and modify to a 2D texture in Metal 3.1 and later, where op is add, and, max, min, or, sub, or xor for int, and uint color type:

Tv atomic_fetch_op(uint2 coord, Tv operand)

Tv atomic_fetch_op(ushort2 coord, Tv operand) const

These member functions execute an atomic min or max to a 2D texture in Metal 3.1 and later:

void atomic_min(uint2 coord, ulong4 operand)

void atomic_min(ushort2 coord, ulong4 operand)

void atomic_max(uint2 coord, ulong4 operand)

void atomic_max(ushort2 coord, ulong4 operand)

6.12.3.1 2D Texture Sampling Example

The following code shows several uses of the 2D texture sample function, depending upon its arguments:

```cpp

texture2d<float> tex;

sampler s;

float2 coord;

int2 offset;

float lod;

// No optional arguments.

float4 clr = tex.sample(s, coord);

// Sample using a mipmap level.

clr = tex.sample(s, coord, level(lod));

// Sample with an offset.

clr = tex.sample(s, coord, offset);

// Sample using a mipmap level and an offset.

clr = tex.sample(s, coord, level(lod), offset);

```

6.12.4 2D Texture Array

For the functions in this section, the following data types and corresponding constructor functions can specify sampling options (lod_options):

bias(float value)

level(float lod)

gradient2d(float2 dPdx, float2 dPdy)

min_lod_clamp(float lod) // All OS: Metal 2.2 and later.

These member functions sample from a 2D texture array:

Tv sample(sampler s, float2 coord, uint array,

    int2 offset = int2(0)) const

Tv sample(sampler s, float2 coord, uint array, lod_options options,

    int2 offset = int2(0)) const

Tv sample(sampler s, float2 coord, uint array, bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

Tv sample(sampler s, float2 coord, uint array,

    gradient2d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

These member functions perform sampler-less reads from a 2D texture array:

Tv read(uint2 coord, uint array, uint lod = 0) const

Tv read(ushort2 coord, ushort array,

    ushort lod = 0) const // All OS: Metal 1.2 and later.

These member functions write to a 2D texture array. In macOS, for Metal 2.2 and earlier, lod must be a zero constant. Metal 2.3 and later lift this restriction for Apple silicon.

void write(Tv color, uint2 coord, uint array, uint lod = 0)

void write(Tv color, ushort2 coord, ushort array,

    ushort lod = 0) // All OS: Metal 1.2 and later.

This member functions gathers four samples for bilinear interpolation when sampling a 2D texture array:

Tv gather(sampler s, float2 coord, uint array,

    int2 offset = int2(0),

    component c = component::x) const

These member functions query a 2D texture array:

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_array_size() const

uint get_num_mip_levels() const

These member functions sample from a sparse 2D texture array in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_sample(sampler s, float2 coord, uint array,

int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord, uint array,

bias options,

int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord, uint array,

level options,

int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord, uint array,

min_lod_clamp min_lod_clamp_options,

int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord, uint array,

bias bias_options,

min_lod_clamp min_lod_clamp_options,

int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord, uint array,

gradient2d options,

int2 offset = int2(0)) const

sparse_color<Tv> sparse_sample(sampler s, float2 coord, uint array,

gradient2d grad_options,

min_lod_clamp min_lod_clamp_options,

int2 offset = int2(0)) const

These functions perform a sampler-less read from a sparse 2D texture array in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_read(ushort2 coord, ushort array,

ushort lod = 0) const

sparse_color<Tv> sparse_read(uint2 coord, uint array,

uint lod = 0) const

This function gathers four samples for bilinear interpolation from a sparse 2D texture array in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather(sampler s, float2 coord, uint array,

int2 offset = int2(0),

component c = component::x) const

These member functions simulate a texture fetch and return the LOD (mip level) computation result in Metal 2.3 and later in iOS, and in Metal 2.2 and later in macOS:

float calculate_clamped_lod(sampler s, float2 coord);

float calculate_unclamped_lod(sampler s, float2 coord);

These member functions execute an atomic load from a 2D texture array in Metal 3.1 and later

Tv atomic_load(uint2 coord, uint array) const

Tv atomic_load(ushort2 coord, ushort array) const

These member functions execute an atomic store to a 2D texture array in Metal 3.1 and later:

void atomic_store(Tv color, uint2 coord, uint array) const

void atomic_store (Tv color, ushort2 coord, ushort array) const

These member functions execute an atomic compare and exchange to a 2D texture array in Metal 3.1 and later:

bool atomic_compare_exchange_weak(uint2 coord, uint array,

thread Tv *expected,

Tv desired) const

bool atomic_compare_exchange_weak(ushort2 coord, ushort array,

thread Tv *expected,

Tv desired) const

These member functions execute an atomic exchange to a 2D texture array in Metal 3.1 and later:

Tv atomic_exchange(uint2 coord, uint array, Tv desired) const

Tv atomic_exchange(ushort2 coord, ushort array, Tv desired) const

These member functions execute an atomic fetch and modify to a 2D texture array in Metal 3.1 and later, where op is add, and, max, min, or, sub, or xor for int, and uint color type:

Tv atomic_fetch_op(uint2 coord, uint array,Tv operand)

Tv atomic_fetch_op(ushort2 coord, ushort array,Tv operand) const

These member functions execute an atomic min or max to a 2D texture array in Metal 3.1 and later:

void atomic_min(uint2 coord, uint array, ulong4 operand)

void atomic_min(ushort2 coord, ushort array, ulong4 operand)

void atomic_max(uint2 coord, uint array, ulong4 operand)

void atomic_max(ushort2 coord, ushort array, ulong4 operand)

## 6.12.5 3D Texture

For the functions in this section, the following data types and corresponding constructor functions can specify sampling options (lod_options):

bias(float value)

level(float lod)

gradient3d(float3 dPdx, float3 dPdy)

min_lod_clamp(float lod) // All OS: Metal 2.2 and later.

These member functions sample from a 3D texture:

Tv sample(sampler s, float3 coord, int3 offset = int3(0)) const

Tv sample(sampler s, float3 coord, lod_options options,

    int3 offset = int3(0)) const

Tv sample(sampler s, float3 coord, bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int3 offset = int3(0)) const

Tv sample(sampler s, float3 coord, gradient3d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int3 offset = int3(0)) const

These member functions perform sampler-less reads from a 3D texture:

Tv read(uint3 coord, uint lod = 0) const

Tv read(ushort3 coord,

    ushort lod = 0) const // All OS: Metal 1.2 and later

These member functions write to a 3D texture. In macOS, in Metal 2.2 and earlier, lod must be a zero constant. Metal 2.3 and later lift this restriction for Apple silicon.

void write(Tv color, uint3 coord, uint lod = 0)

void write(Tv color, ushort3 coord,

    ushort lod = 0) // All OS: Metal 1.2 and later.

These member functions query a 3D texture:

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_depth(uint lod = 0) const

uint get_num_mip_levels() const

These functions sample from a sparse 3D texture in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

    int3 offset = int3(0)) const

```python

sparse_color<Tv> sparse_sample(sampler s, float3 coord, bias options,

    int3 offset = int3(0)) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

    level options,

    int3 offset = int3(0)) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

min_lod_clamp min_lod_clamp_options, int3 offset = int3(0)) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

    bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int3 offset = int3(0)) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

    gradient3d grad_options,

    int3 offset = int3(0)) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

    gradient3d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int3 offset = int3(0)) const

```

These member functions perform a sampler-less read from a sparse 3D texture in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_read(uint3 coord, uint lod = 0) const

sparse_color<Tv> sparse_read(ushort3 coord, ushort lod = 0) const

These member functions simulate a texture fetch and return the LOD (mip level) computation result in Metal 2.3 and later in iOS, and in Metal 2.2 and later in macOS:

float calculate_clamped_lod(sampler s, float3 coord)

float calculate_unclamped_lod(sampler s, float3 coord)

These member functions execute an atomic load from a 3D texture in Metal 3.1 and later: Tv atomic_load(uint3 coord) const Tv atomic_load(ushort3 coord) const

These member functions execute an atomic store to a 3D texture in Metal 3.1 and later: void atomic_store(Tv color, uint3 coord) const

void atomic_store (Tv color, ushort3 coord) const

These member functions execute an atomic compare and exchange to a 3D texture in Metal 3.1 and later:

bool atomic_compare_exchange_weak(uint3 coord, thread Tv *expected,

Tv desired) const

bool atomic_compare_exchange_weak(ushort3 coord,thread Tv *expected,

Tv desired) const

These member functions execute an atomic exchange to a 3D texture in Metal 3.1 and later:

Tv atomic_exchange(uint3 coord, Tv desired) const

Tv atomic_exchange(ushort3 coord, Tv desired) const

These member functions execute an atomic fetch and modify to a 3D texture in Metal 3.1 and later, where op is add, and, max, min, or, sub, or xor for int, and uint color type:

Tv atomic_fetch_op(uint3 coord, Tv operand)

Tv atomic_fetch_op(ushort3 coord, Tv operand) const

These member functions execute an atomic min or max to a 3D texture in Metal 3.1 and later:

void atomic_min(uint3 coord, ulong4 operand)

void atomic_min(ushort3 coord, ulong4 operand)

void atomic_max(uint3 coord, ulong4 operand)

void atomic_max(ushort3 coord, ulong4 operand)

## 6.12.6 Cube Texture

For the functions in this section, the following data types and corresponding constructor functions can specify sampling options (lod_options):

bias(float value)

level(float lod)

gradientcube(float3 dPdx, float3 dPdy)

min_lod_clamp(float lod) // All OS: Metal 2.2 and later.

These member functions sample from a cube texture:

Tv sample(sampler s, float3 coord) const

Tv sample(sampler s, float3 coord, lod_options options) const

Tv sample(sampler s, float3 coord, bias bias_options,

min_lod_clamp min_lod_clamp_options) const

Tv sample(sampler s, float3 coord, gradientcube grad_options,

min_lod_clamp min_lod_clamp_options) const

Table 6.22 describes a cube face and the number used to identify the face.

<div align="center">

Table 6.22. Cube face number

</div>

<table border="1"><tr><td>Face number</td><td>Cube face</td></tr><tr><td>0</td><td>Positive X</td></tr><tr><td>1</td><td>Negative X</td></tr><tr><td>2</td><td>Positive Y</td></tr><tr><td>3</td><td>Negative Y</td></tr><tr><td>4</td><td>Positive Z</td></tr><tr><td>5</td><td>Negative Z</td></tr></table>

This member function gathers four samples for bilinear interpolation when sampling a cube texture:

Tv gather(sampler s, float3 coord, component c = component::x) const

These member functions perform sampler-less reads from a cube texture:

Tv read(uint2 coord, uint face, uint lod = 0) const

Tv read(ushort2 coord, ushort face,

    ushort lod = 0) const // All OS: Metal 1.2 and later.

These member functions write to a cube texture. In macOS, for Metal 2.2 and earlier, lod must be a zero constant. Metal 2.3 and later lift this restriction for Apple silicon.

void write(Tv color, uint2 coord, uint face, uint lod = 0)

void write(Tv color, ushort2 coord, ushort face,

ushort lod = 0) // All OS: Metal 1.2 and later.

These member functions query a cube texture:

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_num_mip_levels() const

These member functions sample from a sparse cube texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_sample(sampler s, float3 coord) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord, bias options)

const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

level options) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

min_lod_clamp min_lod_clamp_options) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

bias bias_options,

min_lod_clamp min_lod_clamp_options) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

gradientcube grad_options) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

gradientcube grad_options,

min_lod_clamp min_lod_clamp_options) const

These member functions perform a sampler-less read from a sparse cube texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_read(ushort2 coord, ushort face, ushort lod = 0) const

sparse_color<Tv> sparse_read(uint2 coord, uint face, uint lod = 0) const

This member function gathers four samples for bilinear interpolation from a sparse cube texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather(sampler s, float3 coord,

component c = component::x) const

These member functions simulate a texture fetch and return the LOD (mip level) computation result in Metal 2.3 and later in iOS, and Metal 2.2 and later in macOS:

float calculate_clamped_lod(sampler s, float3 coord);

float calculate_unclamped_lod(sampler s, float3 coord);

These member functions execute an atomic load from a cube texture in Metal 4 and later: Tv atomic_load(uint2 coord, uint face) const Tv atomic_load(ushort2 coord, ushort face) const

These member functions execute an atomic store to a cube texture in Metal 4 and later:

void atomic_store(Tv color, uint2 coord, uint face) const

void atomic_store (Tv color, ushort2 coord, ushort face) const

These member functions execute an atomic compare and exchange to a cube texture in Metal 4 and later:

bool atomic_compare_exchange_weak(uint2 coord, uint face,

thread Tv *expected,

Tv desired) const

bool atomic_compare_exchange_weak(ushort2 coord, ushort face,

thread Tv *expected,

Tv desired) const

These member functions execute an atomic exchange to a cube texture in Metal 4 and later:

Tv atomic_exchange(uint2 coord, uint face, Tv desired) const

Tv atomic_exchange(ushort2 coord, ushort face, Tv desired) const

These member functions execute an atomic fetch and modify to a cube texture in Metal 4 and later, where op is add, and, max, min, or, sub, or xor for int, and uint color type: Tv atomic_fetch_op(uint2 coord, uint face, Tv operand) Tv atomic_fetch_op(ushort2 coord, ushort face, Tv operand) const

These member functions execute an atomic min or max to a cube texture in Metal 4 and later:

void atomic_min(uint2 coord, uint face, ulong4 operand)

void atomic_min(ushort2 coord, ushort face, ulong4 operand)

void atomic_max(uint2 coord, uint face, ulong4 operand)

void atomic_max(ushort2 coord, ushort face, ulong4 operand)

## 6.12.7 Cube Texture Array

For the functions in this section, the following data types and corresponding constructor functions can specify sampling options (lod_options):

bias(float value)

level(float lod)

gradientcube(float3 dPdx, float3 dPdy)

min_lod_clamp(float lod) // All OS: Metal 2.2 and later.

These member functions sample from a cube texture array:

Tv sample(sampler s, float3 coord, uint array) const

Tv sample(sampler s, float3 coord, uint array,

    lod_options options) const

Tv sample(sampler s, float3 coord, uint array, bias bias_options,

    min_lod_clamp min_lod_clamp_options) const

Tv sample(sampler s, float3 coord, uint array,

    gradientcube grad_options,

    min_lod_clamp min_lod_clamp_options) const

This member function gathers four samples for bilinear interpolation when sampling a cube texture array:

Tv gather(sampler s, float3 coord, uint array,

component c = component::x) const

Tv read(uint2 coord, uint face, uint array, uint lod = 0) const

Tv read(ushort2 coord, ushort face, ushort array,

ushort lod = 0) const // All OS: Metal 1.2 and later.

These member functions write to a cube texture array. In macOS, for Metal 2.2 and earlier, lod must be a zero constant. Metal 2.3 and later lift this restriction for Apple silicon.

void write(Tv color, uint2 coord, uint face, uint array,

uint lod = 0)

void write(Tv color, ushort2 coord, ushort face, ushort array,

ushort lod = 0) // All OS: Metal 1.2 and later.

These member functions query a cube texture array:

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_array_size() const

uint get_num_mip_levels() const

These member functions sample from a sparse cube texture array in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_sample(sampler s, float3 coord,

uint array) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord, uint array,

bias options) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord, uint array,

level options) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord, uint array,

min_lod_clamp min_lod_clamp_options) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord, uint array,

bias bias_options,

min_lod_clamp min_lod_clamp_options) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord, uint array,

gradientcube options) const

sparse_color<Tv> sparse_sample(sampler s, float3 coord, uint array,

gradientcube grad_options,

min_lod_clamp min_lod_clamp_options) const

These member functions perform a sampler-less read from a sparse cube texture array in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_read(ushort2 coord, ushort face,

ushort array, ushort lod = 0) const

sparse_color<Tv> sparse_read(uint2 coord, uint face,

uint array, uint lod = 0) const

This member function gathers four samples for bilinear interpolation from a sparse cube texture array in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather(sampler s, float3 coord, uint array, component c = component::x) const

These member functions simulate a texture fetch and return the LOD (mip level) computation result in Metal 2.3 and later in iOS, and in Metal 2.2 and later in macOS:

float calculate_clamped_lod(sampler s, float3 coord);

float calculate_unclamped_lod(sampler s, float3 coord);

These member functions execute an atomic load from a cube texture array in Metal 4 and later:

Tv atomic_load(uint2 coord, uint face, uint array) const

Tv atomic_load(ushort2 coord, ushort face, ushort array) const

These member functions execute an atomic store to a cube texture array in Metal 4 and later:

void atomic_store(Tv color, uint2 coord, uint face,

uint array) const

void atomic_store (Tv color, ushort2 coord, ushort face,

ushort array) const

These member functions execute an atomic compare and exchange to a cube texture array in Metal 4 and later:

bool atomic_compare_exchange_weak(uint2 coord, uint face,

    uint array,

    thread Tv *expected,

    Tv desired) const

bool atomic_compare_exchange_weak(ushort2 coord, ushort face,

    ushort array,

    thread Tv *expected,

    Tv desired) const

These member functions execute an atomic exchange to a cube texture array in Metal 4 and later:

Tv atomic_exchange(uint2 coord, uint face, uint array,

Tv desired) const

Tv atomic_exchange(ushort2 coord, ushort face, ushort array,

Tv desired) const

These member functions execute an atomic fetch and modify to a cube texture array in Metal 4 and later, where op is add, and, max, min, or, sub, or xor for int, and uint color type:

Tv atomic_fetch_op(uint2 coord, uint face, uint array, Tv operand)

Tv atomic_fetch_op(ushort2 coord, ushort face, ushort array,

Tv operand) const

These member functions execute an atomic min or max to a cube texture array in Metal 4 and later:

void atomic_min(uint2 coord, uint face, uint array, ulong4 operand)

void atomic_min(ushort2 coord, ushort face, ushort array,

ulong4 operand)

void atomic_max(uint2 coord, uint face, uint array, ulong4 operand)

void atomic_max(ushort2 coord, ushort face, ushort array,

ulong4 operand)

6.12.8 2D Multisampled Texture

These member functions perform sampler-less reads from a 2D multisampled texture:

Tv read(uint2 coord, uint sample) const

Tv read(ushort2 coord,

ushort sample) const // All OS: Metal 1.2 and later.

If you have customized sample positions (set with the setSamplePositions:count method of MTLRenderPassDescriptor), then read(coord, sample) returns the data for the sample at the programmed sample position.

These member functions query a 2D multisampled texture:

uint get_width() const

uint get_height() const

uint get_num_samples() const

These member functions perform a sampler-less read from a sparse 2D multisampled texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_read(ushort2 coord, ushort sample) const

sparse_color<Tv> sparse_read(uint2 coord, uint sample) const

## 6.12.9 2D Multisampled Texture Array

macOS: Metal 2 and later support 2D multisampled texture array.

iOS: Metal 2.3 and later support 2D multisampled texture array.

The following member functions can perform sampler-less reads from a 2D multisampled texture array:

Tv read(uint2 coord, uint array, uint sample) const

Tv read(ushort2 coord, ushort array, ushort sample) const

These member functions query a 2D multisampled texture array:

These member functions query a 2D multisampled texture array:

uint get_width() const

uint get_height() const

uint get_num_samples() const

uint get_array_size() const

These functions perform a sampler-less read from a sparse 2D multisampled texture array in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_read(ushort2 coord, ushort array,

    ushort sample) const

sparse_color<Tv> sparse_read(uint2 coord, uint array,

    uint sample) const

## 6.12.10 2D Depth Texture

For the functions in this section, the following data types and corresponding constructor functions can specify sampling options (lod_options):

bias(float value)

level(float lod)

gradient2d(float2 dPdx, float2 dPdy)

min_lod_clamp(float lod) // All OS: Metal 2.2 and later.

T sample(sampler s, float2 coord, int2 offset = int2(0)) const

T sample(sampler s, float2 coord, lod_options options,

    int2 offset = int2(0)) const

T sample(sampler s, float2 coord, bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

T sample(sampler s, float2 coord, gradient2d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

These member functions sample from a 2D depth texture and compare a single component against the comparison value:

T sample_compare(sampler s, float2 coord, float compare_value,

    int2 offset = int2(0)) const

T sample_compare(sampler s, float2 coord, float compare_value,

    lod_options options, int2 offset = int2(0)) const

sample_compare performs a comparison of the compare_value value against the pixel value (1.0 if the comparison passes, and 0.0 if it fails). These comparison result values per pixel are then blended together as in normal texture filtering and the resulting value between 0.0 and 1.0 is returned. In macOS, Metal 2.2 and earlier don't support lod_options values level and min_lod_clamp (the latter, in Metal 2.2 and later); lod must be a zero constant. Metal 2.3 and later lift this restriction for lod_options for Apple silicon.

These member functions perform sampler-less reads from a 2D depth texture:

T read(uint2 coord, uint lod = 0) const

T read(ushort2 coord,

    ushort lod = 0) const // All OS: Metal 1.2 and later.

This built-in function gathers four samples for bilinear interpolation when sampling a 2D depth texture:

Tv gather(sampler s, float2 coord, int2 offset = int2(0)) const

This member function gathers four samples for bilinear interpolation when sampling a 2D depth texture and comparing these samples with a specified comparison value (1.0 if the comparison passes, and 0.0 if it fails):

Tv gather_compare(sampler s, float2 coord, float compare_value,

int2 offset = int2(0)) const

T must be a float type.

The following member functions query a 2D depth texture:

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_num_mip_levels() const

These member functions sample from a sparse 2D depth texture in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

sparse_color<T> sparse_sample(sampler s, float2 coord,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord, bias options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord, level options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord,

    bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord

    gradient2d grad_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord,

    gradient2d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

These member functions sample from a sparse 2D depth texture and compare a single component against a comparison value in Metal 2.2 and later in iOS, and in Metal 2.3 and later in macOS:

```python

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    float compare_value,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    float compare_value,

    bias options,

```

```python

int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    float compare_value,

    level options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    float compare_value,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord

    float compare_value, bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    float compare_value, gradient2d grad_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    float compare_value, gradient2d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

```

These member functions perform a sampler-less read from a sparse 2D depth texture in Metal 2.2 and later, in iOS and Metal 2.3 and later in macOS:

sparse_color<T> sparse_read(ushort2 coord, ushort lod = 0) const

sparse_color<T> sparse_read(uint2 coord, uint lod = 0) const

This member function gathers four samples for bilinear interpolation from a sparse 2D depth texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather(sampler s, float2 coord,

int2 offset = int2(0),

component c = component::x) const

This member function gathers those samples and compares them against a comparison value from a sparse 2D depth texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather_compare(sampler s, float2 coord,

float compare_value,

int2 offset = int2(0)) const

These member functions simulate a texture fetch and return the LOD (mip level) computation result in Metal 2.3 and later in iOS, and Metal 2.2 and later in macOS:

float calculate_clamped_lod(sampler s, float2 coord);

float calculate_unclamped_lod(sampler s, float2 coord);

## 6.12.11 2D Depth Texture Array

The member functions in this section use the following data types and constructor functions to set the sampling option fields of their lod_options parameter:

bias(float value)

level(float lod)

gradient2d(float2 dPdx, float2 dPdy)

min_lod_clamp(float lod) // All OS: Metal 2.2 and later.

These member functions sample from a 2D depth texture array:

T sample(sampler s, float2 coord, uint array,

    int2 offset = int2(0)) const

T sample(sampler s, float2 coord, uint array, lod_options options,

    int2 offset = int2(0)) const

T sample(sampler s, float2 coord, uint array, bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

T sample(sampler s, float2 coord, uint array,

    gradient2d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

These member functions sample from a 2D depth texture array and compare a single component to a value where T is a float type:

T sample_compare(sampler s, float2 coord, uint array,

    float compare_value,int2 offset = int2(0)) const

T sample_compare(sampler s, float2 coord, uint array,

    float compare_value, lod_options options,

    int2 offset = int2(0)) const

The lod_options fields support are:

- level

- bias for all iOS Metal versions and macOS Metal 2.3 and later for Apple silicon

- gradient for iOS Metal versions and macOS Metal 2.3 and later for Apple silicon

- min_lod_clamp for Metal 2.2 and later

- Must be 0 for Metal 2.2 and later

- Can be any value for all iOS Metal versions and macOS Metal 2.3 and later for Apple silicon

These member functions read from a 2D depth texture array without using a sampler:

T read(uint2 coord, uint array, uint lod = 0) const

T read(ushort2 coord, ushort array,

    ushort lod = 0) const // All OS: Metal 1.2 and later.

This member function gathers four samples for bilinear interpolation when sampling a 2D depth texture array:

Tv gather(sampler s, float2 coord, uint array, int2 offset = int2(0)) const

This member function gathers four samples for bilinear interpolation when sampling a 2D depth texture array and compares them to a value where $ \mathsf{T v} $ is a float vector type:

Tv gather_compare(sampler s, float2 coord, uint array,

float compare_value, int2 offset = int2(0)) const

The following member functions query a 2D depth texture array:

```cpp

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_array_size() const

uint get_num_mip_levels() const

```

These member functions sample from a sparse 2D depth texture array, in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_sample(sampler s, float2 coord, uint array,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord, uint array,

    bias options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord, uint array,

    level options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord, uint array,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord, uint array,

    bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord, uint array,

```python

gradient2d grad_options,

int2 offset = int2(0)) const

sparse_color<T> sparse_sample(sampler s, float2 coord, uint array,

gradient2d grad_options,

min_lod_clamp min_lod_clamp_options,

int2 offset = int2(0)) const

```

These functions sample from a sparse 2D depth texture array and compare a single component to a comparison value, in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    uint array, float compare_value,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    uint array, float compare_value,

    bias options, int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    uint array, float compare_value,

    level options, int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    uint array,float compare_value,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    uint array,float compare_value,

    bias bias_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    uint array,

    float compare_value, gradient2d grad_options,

    int2 offset = int2(0)) const

sparse_color<T> sparse_sample_compare(sampler s, float2 coord,

    uint array,float compare_value,

    gradient2d grad_options,

    min_lod_clamp min_lod_clamp_options,

    int2 offset = int2(0)) const

These functions read from a sparse 2D depth texture array without a sampler, in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_read(ushort2 coord, uint array, ushort lod = 0) const

sparse_color<T> sparse_read(uint2 coord, uint array,

uint lod = 0) const

This function gathers four samples for bilinear interpolation from a sparse 2D depth texture array, in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather(sampler s, float2 coord, uint array,

int2 offset = int2(0),

component c = component::x) const

This function gathers those samples and compares them against a value from a sparse 2D depth texture array, in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

float compare_value, int2 offset = int2(0)) const

sparse_color<Tv> sparse_gather_compare(sampler s, float2 coord, uint array,

These functions simulate a texture fetch and return a LOD (mip level) computation result, in Metal 2.3 and later in iOS, and Metal 2.2 and later in macOS:

float calculate_clamped_lod(sampler s, float2 coord);

float calculate_unclamped_lod(sampler s, float2 coord);

## 6.12.12 2D Multisampled Depth Texture

The following member functions can perform sampler-less reads from a 2D multisampled depth texture:

T read(uint2 coord, uint sample) const

T read(ushort2 coord,

    ushort sample) const // All OS: Metal 1.2 and later.

The following member functions query a 2D multisampled depth texture:

The following member functions query a 2D multisampled depth texture.

uint get_width() const

uint get_height() const

uint get_num_samples() const

These member functions perform a sampler-less read from a sparse 2D multisampled depth texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_read(ushort2 coord, ushort sample) const

sparse_color<T> sparse_read(uint2 coord, uint sample) const

## 6.12.13 2D Multisampled Depth Texture Array

macOS: Metal 2 and later support 2D multisampled depth texture array.

iOS: Metal 2.3 and later support 2D multisampled depth texture array.

The following member functions perform sampler-less reads from a 2D multisampled depth texture array:

Tv read(uint2 coord, uint array, uint lod = 0) const

Tv read(ushort2 coord, ushort array, ushort lod = 0) const

The following member functions query a 2D multisampled depth texture array:

```cpp

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_array_size() const

```

These member functions perform a sampler-less read from a sparse 2D multisampled depth texture array in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_read(ushort2 coord, ushort array,

ushort sample) const

sparse_color<T> sparse_read(uint2 coord, uint array,

uint sample) const

## 6.12.14 Cube Depth Texture

For the functions in this section, the following data types and corresponding constructor functions can specify sampling options (lod_options):

bias(float value)

level(float lod)

gradientcube(float3 dPdx, float3 dPdy)

min_lod_clamp(float lod) // All OS: Metal 2.2 and later.

The following member functions sample from a cube depth texture:

T sample(sampler s, float3 coord) const

T sample(sampler s, float3 coord, lod_options options) const

T sample(sampler s, float3 coord, bias bias_options,

    min_lod_clamp min_lod_clamp_options) const

T sample(sampler s, float3 coord, gradientcube grad_options,

    min_lod_clamp min_lod_clamp_options) const

The following member functions sample from a cube depth texture and compare a single component against the specified comparison value:

T sample_compare(sampler s, float3 coord, float compare_value) const

T sample_compare(sampler s, float3 coord, float compare_value,

    lod_options options) const

T must be a float type. In macOS, Metal 2.2 and earlier support lod_options values level and min_lod_clamp (the latter, in Metal 2.2 and later), and lod must be a zero constant. Metal 2.3 and later lift this restriction for lod_options for Apple silicon.

The following member functions perform sampler-less reads from a cube depth texture:

T read(uint2 coord, uint face, uint lod = 0) const

T read(ushort2 coord, ushort face,

    ushort lod = 0) const // All OS: Metal 1.2 and later.

This member function gathers four samples for bilinear interpolation when sampling a cube depth texture:

Tv gather(sampler s, float3 coord) const

This member function gathers four samples for bilinear interpolation when sampling a cube texture and comparing these samples with a specified comparison value:

Tv gather_compare(sampler s, float3 coord, float compare_value) const

T must be a float type.

The following member functions query a cube depth texture:

```cpp

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_num_mip_levels() const

```

These member functions sample from a sparse cube depth texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_sample(sampler s, float3 coord) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

bias options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

level options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

min_lod_clamp min_lod_clamp_options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

bias bias_options,

min_lod_clamp min_lod_clamp_options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

gradientcube grad_options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

gradientcube grad_options,

min_lod_clamp min_lod_clamp_options) const

These member functions sample from a sparse cube depth texture and compare a single component against a comparison value in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

float compare_value) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

float compare_value, bias options) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

float compare_value, level options) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

float compare_value,

min_lod_clamp min_lod_clamp_options) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

float compare_value, bias bias_options,

min_lod_clamp min_lod_clamp_options) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

float compare_value,

gradient2d grad_options) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

float compare_value, gradient2d grad_options,

min_lod_clamp min_lod_clamp_options) const

These member functions perform a sampler-less read from a sparse cube depth texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_read(ushort2 coord, ushort face

    ushort lod = 0) const

sparse_color<T> sparse_read(uint2 coord, uint face,

    uint lod = 0) const

This member function gathers four samples for bilinear interpolation from a sparse cube depth texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather(sampler s, float3 coord) const

This member function gathers those samples and compare them against a comparison value from a sparse cube depth texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather_compare(sampler s, float3 coord, float compare_value) const

These member functions simulate a texture fetch and return the LOD (mip level) computation result in Metal 2.3 and later in iOS, and Metal 2.2 and later in macOS:

float calculate_clamped_lod(sampler s, float3 coord);

float calculate_unclamped_lod(sampler s, float3 coord);

## 6.12.15 Cube Depth Texture Array

For the functions in this section, the following data types and corresponding constructor functions can specify sampling options (lod_options):

bias(float value)

level(float lod)

gradientcube(float3 dPdx, float3 dPdy)

min_lod_clamp(float lod) // All OS: Metal 2.2 and later.

These member functions sample from a cube depth texture array:

T sample(sampler s, float3 coord, uint array) const

T sample(sampler s, float3 coord, uint array,

    lod_options options) const

T sample(sampler s, float3 coord, uint array, bias bias_options,

    min_lod_clamp min_lod_clamp_options) const

T sample(sampler s, float3 coord, uint array,

    gradientcube grad_options,

    min_lod_clamp min_lod_clamp_options) const

These member functions sample from a cube depth texture and compare a single component against the specified comparison value:

T sample_compare(sampler s, float3 coord, uint array, float compare_value) const

T sample_compare(sampler s, float3 coord, uint array, float compare_value, lod_options options) const

T must be a float type. In macOS, Metal 2.2 and earlier support lod_options values level and min_lod_clamp (the latter, in Metal 2.2 and later), and lod must be a zero constant. Metal 2.3 and later lift this restriction for lod_options for Apple silicon.

These member functions perform sampler-less reads from a cube depth texture array:

T read(uint2 coord, uint face, uint array, uint lod = 0) const

T read(ushort2 coord, ushort face, ushort array,

    ushort lod = 0) const // All OS: Metal 1.2 and later.

This member function gathers four samples for bilinear interpolation when sampling a cube depth texture:

Tv gather(sampler s, float3 coord, uint array) const

This member function gathers four samples for bilinear interpolation when sampling a cube depth texture and comparing these samples with a specified comparison value:

Tv gather_compare(sampler s, float3 coord, uint array, float compare_value) const

T must be a float type.

These member functions query a cube depth texture:

uint get_width(uint lod = 0) const

uint get_height(uint lod = 0) const

uint get_array_size() const

uint get_num_mip_levels() const

These member functions sample from a sparse cube depth texture array in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_sample(sampler s, float3 coord,

    uint array) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

    uint array, bias options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

    uint array, level options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

    uint array,

    min_lod_clamp min_lod_clamp_options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

    uint array, bias bias_options,

```python

min_lod_clamp min_lod_clamp_options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

    uint array,

    gradientcube grad_options) const

sparse_color<T> sparse_sample(sampler s, float3 coord,

    uint array,

    gradientcube grad_options,

    min_lod_clamp min_lod_clamp_options) const

```

These member functions sample from a sparse cube depth texture array and compare a single component against a comparison value in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

    uint array, float compare_value) const

sparse_color<T> sparse_sample_compare(sampler s,float3 coord,

    uint array, float compare_value,

    bias options) const

sparse_color<T> sparse_sample_compare(sampler s,float3 coord,

    uint array, float compare_value,

    level options) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

    uint array, float compare_value,

    min_lod_clamp min_lod_clamp_options) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

    uint array, float compare_value,

    bias bias_options,

    min_lod_clamp min_lod_clamp_options) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

    uint array,float compare_value,

    gradient2d grad_options) const

sparse_color<T> sparse_sample_compare(sampler s, float3 coord,

    uint array, float compare_value,

    gradient2d grad_options,

    min_lod_clamp min_lod_clamp_options) const

These member functions perform a sampler-less read from a sparse cube depth texture array in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<T> sparse_read(ushort2 coord, ushort face, ushort array,

ushort lod = 0) const

sparse_color<T> sparse_read(uint2 coord, uint face, uint array,

uint lod = 0) const

This member function gathers four samples for bilinear interpolation from a sparse cube depth texture array in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather(sampler s, float3 coord, uint array) const

This member function gathers those samples and compare them against a comparison value from a sparse 2D depth texture in Metal 2.2 and later in iOS, and Metal 2.3 and later in macOS:

sparse_color<Tv> sparse_gather_compare(sampler s, float3 coord,

uint array,

float compare_value) const

These member functions simulate a texture fetch and return the LOD (mip level) computation result in Metal 2.3 and later in iOS, and Metal 2.2 and later in macOS:

float calculate_clamped_lod(sampler s, float3 coord);

float calculate_unclamped_lod(sampler s, float3 coord);

## 6.12.16 Texture Buffer Functions

All OS: Metal 2.1 and later support texture buffers and these functions.

The following member functions can read from and write to an element in a texture buffer (also see section 2.9.1):

Tv read(uint coord) const;

void write(Tv color, uint coord);

These member functions execute an atomic load from a texture buffer in Metal 3.1 and later: Tv atomic_load(uint coord) const Tv atomic_load(ushort coord) const

These member functions execute an atomic store to a texture buffer in Metal 3.1 and later:

void atomic_store(Tv color, uint coord) const

void atomic_store (Tv color, ushort coord) const

These member functions execute an atomic compare and exchange to a texture buffer in Metal 3.1 and later:

bool atomic_compare_exchange_weak(uint coord, thread Tv *expected,

Tv desired) const

bool atomic_compare_exchange_weak(ushort coord, thread Tv *expected,

Tv desired) const

These member functions execute an atomic exchange to a texture buffer in Metal 3.1 and later: Tv atomic_exchange(uint coord, Tv desired) const Tv atomic_exchange(ushort coord, Tv desired) const

These member functions execute an atomic fetch and modify to a texture buffer in Metal 3.1 and later, where op is add, and, max, min, or, sub, or xor for int, and uint color type: Tv atomic_fetch_op(uint coord, Tv operand) Tv atomic_fetch_op(ushort coord, Tv operand) const

These member functions execute an atomic min or max to a texture buffer in Metal 3.1 and later:

```cpp

void atomic_min(uint coord, ulong4 operand)

void atomic_min(ushort coord, ulong4 operand)

void atomic_max(uint coord, ulong4 operand)

void atomic_max(ushort coord, ulong4 operand)

```

The following example uses the read method to access a texture buffer:

```cpp

kernel void

myKernel(texture_buffer<float, access::read> myBuffer)

{

    uint index = ...;

    float4 value = myBuffer.read(index);

}

```

Use the following method to query the number of elements in a texture buffer:

uint get_width() const;

## 6.12.17 Texture Synchronization Functions

All OS: Metal 1.2 and later support texture synchronization functions.

The texture fence() member function ensures that writes to the texture by a thread become visible to subsequent reads from that texture by the same thread (the thread that is performing the write). Texture types (including texture buffers) that you can declare with the access::read_write attribute support the Fence function.

void fence()

The following example shows how to use a texture fence function to make sure that writes to a texture by a thread are visible to later reads to the same location by the same thread:

```cpp

kernel void

my_kernel(texture2d<float, access::read_write> texA,

        ...

        ushort2 gid [[thread_position_in_grid]])

{

    float4 clr = ...;

    texA.write(clr, gid);

    ...

    // Use fence to ensure that writes by thread are

    // visible to later reads by the thread.

    texA.fence();

    clr_new = texA.read(gid);

    ...

}

```

## 6.12.18 Null Texture Functions

All OS: Metal 1.2 and later support null texture functions.

macOS: Metal 2 and later support null texture functions for texture2d_ms_array and depth2d_ms_array.

Use the following functions to determine if a texture is a null texture. If the texture is a null texture, is_null_texture returns true; otherwise, return false:

bool is_null_texture(texture1d<T, access>);

bool is_null_texture(texture1d_array<T, access>);

bool is_null_texture(texture2d<T, access>);

bool is_null_texture(texture2d_array<T, access>);

bool is_null_texture(texture3d<T, access>);

bool is_null_texture(texturecube<T, access>);

bool is_null_texture(texturecube_array<T, access>);

bool is_null_texture(texture2d_ms<T, access>);

// Metal 2 and later support texture2d_ms_array in macOS, and

// Metal 2.3 and later in iOS.

bool is_null_texture(texture2d_ms_array<T, access>);

bool is_null_texture(depth2d<T, access>);

bool is_null_texture(depth2d_array<T, access>);

bool is_null_texture(depthcube<T, access>);

bool is_null_texture(depthcube_array<T, access>);

bool is_null_texture(depth2d_ms<T, access>);

// depth2d_ms_array is macOS only, in Metal 2 and later.

bool is_null_texture(depth2d_ms_array<T, access>);

The behavior of calling any texture member function with a null texture is undefined.

## 6.13 Imageblock Functions

macOS: Metal 2.3 and later support imageblocks for Apple silicon.

iOS: Metal 2 and later support imageblocks.

This section lists the Metal member functions for imageblocks. (For more about the imageblock data type, see sections 2.11 and 5.6.)

The following member functions query information about the imageblock:

```python

ushort get_width() const;

ushort get_height() const;

ushort get_num_samples() const;

```

Use the following member function to query the number of unique color entries for a specific location given by an (x,y) coordinate inside the imageblock:

ushort get_num_colors(ushort2 coord) const;

The following member function returns the color coverage mask (that is, whether a given color covers one or more samples in the imageblock). Each sample is identified by its bit position in the return value. If a bit is set, then this indicates that this sample uses the color index.

ushort get_color_coverage_mask(ushort2 coord, ushort color_index) const;

color_index is a value from 0 to get_num_colors() - 1.

## 6.13.1 Functions for Imageblocks with Implicit Layout

Use the following functions to read or write an imageblock at pixel rate for a given (x, y) coordinate inside the imageblock:

T read(ushort2 coord) const;

void write(T data, ushort2 coord);

Use the following member function to read or write an imageblock at sample or color rate. coord specifies the (x,y) coordinate inside the imageblock, and index is the sample or color index.

enum class imageblock_data_rate { color, sample };

```cpp

T read(ushort2 coord, ushort index,

       imageblock_data_rate data_rate) const;

void write(T data, ushort2 coord, ushort index,

       imageblock_data_rate data_rate);

Example:

struct Foo {

    float4 a [[color(0)]];

    int4 b [[color(1)]];

};

kernel void

my_kernel(imageblock<Foo, imageblock_layout_implicit> img_blk,

           ushort2 lid [[thread_position_in_threadgroup]] ...)

{

    ...

    Foo f = img_blk.read(lid); float4 r = f.a;

    ...

    f.a = r;

    ...

    img_blk.write(f, lid);

}

```

Use the following member function to write an imageblock with a color coverage mask. You must use this member function when writing to an imageblock at color rate:

void write(T data, ushort2 coord, ushort color_coverage_mask);

Use the following member functions to get a region of a slice for a given data member in the imageblock. You use these functions to write data associated with a specific data member described in the imageblock for all threads in the threadgroup to a specified region in a texture. color_index refers to the data member declared in the structure type specified in imageblock<T> with the [[color(n)]] attribute where n Is color_index. size is the actual size of the copied slice.

```javascript

const imageblock_slice<E, imageblock_layout_implicit> slice(ushort

color_index) const;

const imageblock_slice<E, imageblock_layout_implicit> slice(ushort

color_index, ushort2 size) const;

```

The region to copy has an origin of (0,0). The slice(...) member function that does not have the argument size copies the entire width and height of the imageblock.

## 6.13.2 Functions for Imageblocks with Explicit Layout

Use the following member functions to get a reference to the imageblock data for a specific location given by an (x,y) coordinate inside the imageblock. Use these member functions when reading or writing data members in an imageblock At pixel rate.

threadgroup_imageblock T* data(ushort2 coord);

const threadgroup_imageblock T* data(ushort2 coord) const;

Use the following member functions to get a reference to the imageblock data for a specific location given by an (x,y) coordinate inside the imageblock and a sample or color index. Use these member functions when reading or writing data members in an imageblock at sample or color rate. T is the type specific in the imageblock<T> templated declaration. coord is the coordinate in the imageblock, and index is the sample or color index for a multisampled imageblock. data_rate specifies whether the index is a color or sample index. If coord refers to a location outside the imageblock dimensions or if index is an invalid index, the behavior of data() is undefined.

```cpp

enum class imageblock_data_rate { color, sample };

threadgroup_imageblock T* data(ushort2 coord, ushort index,

imageblock_data_rate data_rate);

const threadgroup_imageblock T* data(ushort2 coord, ushort index,

imageblock_data_rate data_rate) const;

```

Calling the data(coord) member function for an imageblock that stores pixels at sample or color rate is equivalent to calling data(coord, 0, imageblock_data_rate::sample).

Example:

```c

struct Foo {

    rgba8unorm<half4> a;

    int b;

};

kernel void

my_kernel(imageblock<Foo> img_blk,

            ushort2 lid [[thread_position_in_threadgroup]] ...)

{

    ...

    threadgroup_imageblock Foo* f = img_blk.data(lid);

    half4 r = f->a;

    f->a = r;

    ...

}

```

Use the following write member function to write an imageblock with a color coverage mask. You must use this member function when writing to an imageblock at color rate.

void write(T data, ushort2 coord, ushort color_coverage_mask);

Use the following slice member functions to get a region of a slice for a given data member in the imageblock structure. You use this function to write data associated with a specific data member described in the imageblock structure for all threads in the threadgroup to a specified region in a texture.

data_member is a data member declared in the structure type specified in imageblock<T>. size is the actual size of the copied slice.

```javascript

const imageblock_slice<E, imageblock_layout_explicit>

slice(const threadgroup_imageblock E& data_member) const;

const imageblock_slice<E, imageblock_layout_explicit>

slice(const threadgroup_imageblock E& data_member, ushort2 size)

const;

```

The region to copy has an origin of (0,0). The slice (...) member function that doesn't have the argument size copies the entire width and height of the imageblock.

## 6.13.3 Writing an Imageblock Slice to a Region in a Texture

Use the following write(...) member function in these texture types to write pixels associated with a slice in the imageblock to a texture starting at a location that coord provides.

A write to a texture from an imageblock is out-of-bounds if, and only if, it meets any of these conditions:

- The accessed coordinates are out-of-bounds.

- The level of detail argument is out-of-bounds.

- Any part of the imageblock_slice accesses outside the texture.

An out-of-bounds write to a texture is undefined. Note that the write from imageblock_slice to a texture must have matching MSAA modes or the result is undefined.

## For a 1D texture:

```cpp

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    uint coord, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    ushort coord, ushort lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    uint coord, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    ushort coord, ushort lod = 0);

```

## For a 1D texture array:

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

uint coord, uint array, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

ushort coord, ushort array, ushort lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

uint coord, uint array, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

ushort coord, ushort array, ushort lod = 0);

## For a 2D texture:

```cpp

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    uint2 coord, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    ushort2 coord, ushort lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    uint2 coord, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    ushort2 coord, ushort lod = 0);

```

## For a 2D MSAA texture:

```cpp

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    uint2 coord, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    ushort2 coord, ushort lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    uint2 coord, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    ushort2 coord, ushort lod = 0);

```

## For a 2D texture array:

```cpp

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    uint2 coord, uint array, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    ushort2 coord, ushort array, ushort lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    uint2 coord, uint array, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    ushort2 coord, ushort array, ushort lod = 0);

```

## For a cube texture:

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

uint2 coord, uint face, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

ushort2 coord, ushort face, ushort lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

uint2 coord, uint face, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

ushort2 coord, ushort face, ushort lod = 0);

## For a cube texture array:

```cpp

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    uint2 coord, uint face, uint array, uint lod =

0);

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

    ushort2 coord, ushort face, ushort array, ushort

lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    uint2 coord, uint face, uint array, uint lod =

0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

    ushort2 coord, ushort face, ushort array, ushort

lod = 0);

```

## For a 3D texture:

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

uint3 coord, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_explicit> slice,

ushort3 coord, ushort lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

uint3 coord, uint lod = 0);

void write(imageblock_slice<E, imageblock_layout_implicit> slice,

ushort3 coord, ushort lod = 0);

## Example:

```c

struct Foo {

    half4 a;

    int b;

    float c;

};

kernel void

my_kernel(texture2d<half> src [[ texture(0) ]],

```

```cpp

        texture2d<half, access::write> dst [[ texture(1) ]],

        imageblock<Foo> img_blk,

        ushort2 lid [[ thread_position_in_threadgroup ]],

        ushort2 gid [[ thread_position_in_grid ]])

    // Read the pixel from the input image using the thread ID.

    half4 clr = src.read(gid);

    // Get the image slice.

    threadgroup_imageblock Foo* f = img_blk.data(lid);

    // Write the pixel in the imageblock using the thread ID in

    // threadgroup.

    f->a = clr;

    // A barrier to make sure all threads finish writing to the

    // imageblock.

    //

    // In this case, each thread writes to its location in the

    // imageblock so a barrier isn't necessary.

    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);

    // Process the pixels in imageblock, and update the elements in

    // slice.

    process_pixels_in_imageblock(img_blk, gid, lid);

    // A barrier to make sure all threads finish writing to the

    // elements in the imageblock.

    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);

    // Write a specific element in an imageblock to the output

    // image. Only one thread in the threadgroup performs the

    // imageblock write.

    if (lid.x == 0 && lid.y == 0)

        dst.write(img_blk.slice(f->a), gid);

}

```

## 6.14 Pack and Unpack Functions

This section lists the Metal functions, defined in the header <metal_pack>, for converting a vector floating-point data to and from a packed integer value. Refer to subsections of section 8.7 for details on how to convert from an 8-, 10-, or 16-bit signed or unsigned integer value to a normalized single- or half-precision floating-point value and vice-versa.

## 6.14.1 Unpack and Convert Integers to a Floating-Point Vector

Table 6.23 lists functions that unpack multiple values from a single unsigned integer and then converts them into floating-point values that are stored in a vector.

<div align="center">

Table 6.23. Unpack functions

</div>

<table border="1"><tr><td>Built-in unpack functions</td><td>Description</td></tr><tr><td>float4 unpack_unorm4x8_to_float(uint x)
float4 unpack_snorm4x8_to_float(uint x)
half4 unpack_unorm4x8_to_half(uint x)
half4 unpack_snorm4x8_to_half(uint x)</td><td>Unpack a 32-bit unsigned integer into four 8-bit signed or unsigned integers and then convert each 8-bit signed or unsigned integer value to a normalized single- or half-precision floating-point value to generate a 4-component vector.</td></tr><tr><td>float4
unpack_unorm4x8_srgb_to_float(uint x)
half4 unpack_unorm4x8_srgb_to_half(uint x)</td><td>Unpack a 32-bit unsigned integer into four 8-bit signed or unsigned integers and then convert each 8-bit signed or unsigned integer value to a normalized single- or half-precision floating-point value to generate a 4-component vector. The r, g, and b color values are converted from sRGB to linear RGB.</td></tr><tr><td>float2 unpack_unorm2x16_to_float(uint x)
float2 unpack_snorm2x16_to_float(uint x)
half2 unpack_unorm2x16_to_half(uint x)
half2 unpack_snorm2x16_to_half(uint x)</td><td>Unpack a 32-bit unsigned integer into two 16-bit signed or unsigned integers and then convert each 16-bit signed or unsigned integer value to a normalized single- or half-precision floating-point value to generate a 2-component vector.</td></tr><tr><td>float4 unpack_unorm10a2_to_float(uint x)
float3 unpack_unorm565_to_float(ushort x)
half4 unpack_unorm10a2_to_half(uint x)
half3 unpack_unorm565_to_half(ushort x)</td><td>Convert a 10a2 (1010102) or 565 color value to the corresponding normalized single- or half-precision floating-point vector.</td></tr><tr><td>float4 unpack_snorm10a2_to_float(uint x)
half4 unpack_snorm10a2_to_half(uint x)
All OS: Metal 4 and later</td><td>Convert a 10a2 (1010102) signed color value to the corresponding normalized single- or half-precision floating-point vector.</td></tr></table>

When converting from a 16-bit unsigned normalized or signed normalized value to a halfprecision floating-point, the unpack_unorm2x16_to_half and unpack_snorm2x16_to_half functions may lose precision.

## 6.14.2 Convert Floating-Point Vector to Integers, then Pack the Integers

Table 6.24 lists functions that start with a floating-point vector, converts the components into integer values, and then packs the multiple values into a single unsigned integer.

<div align="center">

Table 6.24. Pack functions

</div>

<table border="1"><tr><td>Built-in pack functions</td><td>Description</td></tr><tr><td>uint pack_float_to_unorm4x8(float4 x)
uint pack_float_to_snorm4x8(float4 x)
uint pack_half_to_unorm4x8(half4 x)
uint pack_half_to_snorm4x8(half4 x)</td><td>Convert a four-component vector normalized single- or half-precision floating-point value to four 8-bit integer values and pack these 8-bit integer values into a 32-bit unsigned integer.</td></tr><tr><td>uint pack_float_to_srgb_unorm4x8(float4 x)
uint pack_half_to_srgb_unorm4x8(half4 x)</td><td>Convert a four-component vector normalized single- or half-precision floating-point value to four 8-bit integer values and pack these 8-bit integer values into a 32-bit unsigned integer. The color values are converted from linear RGB to sRGB.</td></tr><tr><td>uint pack_float_to_unorm2x16(float2 x)
uint pack_float_to_snorm2x16(float2 x)
uint pack_half_to_unorm2x16(half2 x)
uint pack_half_to_snorm2x16(half2 x)</td><td>Convert a two-component vector of normalized single- or half-precision floating-point values to two 16-bit integer values and pack these 16-bit integer values into a 32-bit unsigned integer.</td></tr><tr><td>uint pack_float_to_unorm10a2(float4)
ushort pack_float_to_unorm565(float3)
uint pack_half_to_unorm10a2(half4)
ushort pack_half_to_unorm565(half3)</td><td>Convert a three- or four-component vector of normalized single- or half-precision floating-point values to a packed,10a2(1010102) or565color integer value.</td></tr><tr><td>uint pack_float_to_snorm10a2(float4)
uint pack_half_to_snorm10a2(half4)
All OS: Metal 4 and later.</td><td>Convert a four-component vector of normalized single- or half-precision floating-point values to a packed10a2(1010102) signed color integer value.</td></tr></table>

## 6.15 Atomic Functions

The Metal programming language implements a subset of the C++17 atomics and synchronization operations. Metal atomic functions must operate on Metal atomic data, as described in section 2.6.

Atomic operations play a special role in making assignments in one thread visible to another thread. A synchronization operation on one or more memory locations is either an acquire operation, a release operation, or both. A synchronization operation without an associated memory location is a fence and can be either an acquire fence, a release fence, or both. In addition, there are relaxed atomic operations that are not synchronization operations.

There are only a few kinds of operations on atomic types, although there are many instances of those kinds. This section specifies each general kind.

Atomic functions are defined in the header <metal_atomic>.

## 6.15.1 Memory Order

The enumeration memory_order specifies the detailed regular (nonatomic) memory synchronization operations (see section 29.3 of the C++17 specification) and may provide for operation ordering:

```c

enum memory_order {

    memory_order_relaxed,

    memory_order_seq_cst

};

```

For atomic operations other than atomic_thread_fence, memory_order_relaxed is the only enumeration value. With memory_order_relaxed, there are no synchronization or ordering constraints; the operation only requires atomicity. These operations do not order memory, but they guarantee atomicity and modification order consistency. A typical use for relaxed memory ordering is updating counters, such as reference counters because this only requires atomicity, but neither ordering nor synchronization.

In Metal 3.2 and later, you can use memory_order_seq_cst on atomic_thread_fence to indicate that everything that happens before a store operation in one thread becomes a visible side effect in the thread that performs the load, and establishes a single total modification order of all tagged atomic operations.

## 6.15.2 Thread Scope

All OS: Metal 3.2 and later support thread_scope for Apple silicon.

The enumeration thread_scope denotes a set of threads for the memory order constraint that the memory_order provides:

```c

enum thread_scope {

    thread_scope_thread,

    thread_scope_simdgroup,

    thread_scope_threadgroup,

```

```c

    thread_scope_device

}

```

Informally, the thread scope on a synchronization operation defines the set of threads with which this operation may synchronize, or which may synchronize with the operation. You use it with atomic_thread_fence.

## 6.15.3 Fence Functions

All OS: Metal 3.2 and later support atomic_thread_fence for Apple silicon.

The atomic_thread_fence establishes memory synchronization ordering of nonatomic and relaxed atomic accesses, according to the memory order and thread scope, without an associated atomic function:

void atomic_thread_fence(mem_flags flags, memory_order order,

thread_scope scope = thread_scope_device)

A fence operates on the following address space scopes:

- threadgroup, if mem_flags include mem_threadgroup

- threadgroup_imageblock, if mem_flags include mem_threadgroup_imageblock

- object_data, if mem_flags include mem_object_data

- device, if mem_flags include mem_device

- texture, if mem_flags include mem_texture

A fence accepts a scope parameter (see section 6.15.2) that denotes the set of threads for the fence that the order affects. Depending on the value of order (see section 6.15.1), this operation:

- has no effects, if order == memory_order_relaxed

- is a sequentially consistent acquire and release fence, if order == memory_order_seq_cst

An atomic_thread_fence imposes different synchronization constraints than an atomic store operation with the same memory_order. An atomic store-release operation prevents all preceding writes from moving past the store-release, and an atomic_thread_fence with memory_order_seq_cst ordering prevents all preceding writes from moving past all subsequent stores within that scope.

## 6.15.4 Atomic Functions

In addition, accesses to atomic objects may establish interthread synchronization and order nonatomic memory accesses as specified by memory_order.

In the atomic functions described in the subsections of this section:

- A refers to one of the atomic types.

- C refers to its corresponding nonatomic type.

- M refers to the type of the other argument for arithmetic operations. For atomic integer types, M is C.

Note that each atomic function may support only some types. The following sections indicate which type A Metal supports.

All OS: Metal 1 and later support functions with names that end with _explicit (such as atomic_store_explicit or atomic_load_explicit) unless otherwise indicated. Metal 3 supports the atomic_float for device memory only.

iOS: Metal 2 and later support the atomic_store, atomic_load, atomic_exchange, atomic_compare_exchange_weak, and atomic_fetch_key functions.

## 6.15.4.1 Atomic Store Functions

These functions atomically replace the value pointed to by object with desired. These functions support atomic types A of atomic_int, atomic_uint, atomic_bool, and atomic_float. Atomic store supports atomic_float only for device memory.

All OS: Support for the atomic_store_explicit function with memory_order_relaxed supported, as indicated.

```cpp

void atomic_store_explicit(threadgroup A* object, C desired,

    memory_order order) // All OS: Since Metal 2.

void atomic_store_explicit(volatile threadgroup A* object,

    C desired,

    memory_order order) // All OS: Since Metal 1.

void atomic_store_explicit(device A* object, C desired,

    memory_order order) // All OS: Since Metal 2.

void atomic_store_explicit(volatile device A* object, C desired,

    memory_order order) // All OS: Since Metal 1.

```

## 6.15.4.2 Atomic Load Functions

These functions atomically obtain the value pointed to by object. These functions support atomic types A of atomic_int, atomic_uint, atomic_bool, and atomic_float. Atomic load supports atomic_float only for device memory.

All OS: Support for the atomic_load_explicit function with memory_order_relaxed supported, as indicated.

C atomic_load_explicit(const threadgroup A* object,

memory_order order) // All OS: Since Metal 2.

C atomic_load_explicit(const volatile threadgroup A* object,

memory_order order) // All OS: Since Metal 1.

C atomic_load_explicit(const device A* object,

memory_order order) // All OS: Since Metal 2.

C atomic_load_explicit(const volatile device A* object,

memory_order order) // All OS: Since Metal 1.

## 6.15.4.3 Atomic Exchange Functions

These functions atomically replace the value pointed to by object with desired and return the value object previously held. These functions support atomic types A of atomic_int, atomic_uint, atomic_bool, and atomic_float.

All OS: Support for the atomic_exchange_explicit function with memory_order_relaxed supported, as indicated.

C atomic_exchange_explicit(threadgroup A* object,

C desired,

memory_order order) // All OS: Since Metal 2.

C atomic_exchange_explicit(volatile threadgroup A* object,

C desired,

memory_order order) // All OS: Since Metal 1.

C atomic_exchange_explicit(device A* object,

C desired,

memory_order order) // All OS: Since Metal 2.

C atomic_exchange_explicit(volatile device A* object,

C desired,

memory_order order) // All OS: Since Metal 1.

## 6.15.4.4 Atomic Compare and Exchange Functions

These compare-and-exchange functions atomically compare the value in *object with the value in *expected. If those values are equal, the compare-and-exchange function performs a read-modify-write operation to replace *object with desired. Otherwise if those values are not equal, the compare-and-exchange function loads the actual value from *object into *expected. If the underlying atomic value in *object was successfully changed, the compare-and-exchange function returns true; otherwise it returns false. These functions support atomic types A of atomic_int, atomic_uint, atomic_bool, and atomic_float.

Copying is performed in a manner similar to std::memcpy. The effect of a compare-and exchange function is:

```javascript

if (memcmp(object, expected, sizeof(*object)) == 0) {

    memcpy(object, &desired, sizeof(*object));

} else {

    memcpy(expected, object, sizeof(*object));

}

```

All OS: Support for the atomic_compare_exchange_weak_explicit function supported as indicated; support for memory_order_relaxed for indicating success and failure. If the comparison is true, the value of success affects memory access, and if the comparison is false, the value of failure affects memory access.

bool atomic_compare_exchange_weak_explicit(threadgroup A* object,

    C *expected, C desired, memory_order success,

    memory_order failure) // All OS: Since Metal 2.

bool atomic_compare_exchange_weak_explicit(volatile threadgroup A* object,

    C *expected, C desired, memory_order success,

    memory_order failure) // All OS: Since Metal 1.

bool atomic_compare_exchange_weak_explicit(device A* object,

    C *expected, C desired, memory_order success,

    memory_order failure) // All OS: Since Metal 2.

bool atomic_compare_exchange_weak_explicit(volatile device A* object,

    C *expected, C desired, memory_order success,

    memory_order failure) // All OS: Since Metal 1.

## 6.15.4.5 Atomic Fetch and Modify Functions

All OS: The following atomic fetch and modify functions are supported, as indicated.

The only supported value for order is memory_order_relaxed.

C atomic_fetch_key_explicit(threadgroup A* object,

M operand,

memory_order order) // All OS: Since Metal 2.

C atomic_fetch_key_explicit(volatile threadgroup A* object,

M operand,

memory_order order) // All OS: Since Metal 1.

C atomic_fetch_key_explicit(device A* object,

M operand,

memory_order order) // All OS: Since Metal 2.

C atomic_fetch_key_explicit(volatile device A* object,

M operand,

memory_order order) // All OS: Since Metal 1.

The key in the function name is a placeholder for an operation name listed in the first column of Table 6.25, such as atomic_fetch_add_explicit. The operations detailed in Table 6.25 are arithmetic and bitwise computations. The function atomically replaces the value pointed to by object with the result of the specified computation (third column of Table 6.25). The function returns the value that object held previously. There are no undefined results.

These functions are applicable to any atomic object of type atomic_int, and atomic_uint. Atomic add and sub support atomic_float only in device memory.

<div align="center">

Table 6.25. Atomic operations

</div>

<table border="1"><tr><td>Key</td><td>Operator</td><td>Computation</td></tr><tr><td>add</td><td>+</td><td>Addition</td></tr><tr><td>and</td><td>&amp;</td><td>Bitwise and</td></tr><tr><td>max</td><td>max</td><td>Compute max</td></tr><tr><td>min</td><td>min</td><td>Compute min</td></tr><tr><td>or</td><td>|</td><td>Bitwise inclusive or</td></tr><tr><td>sub</td><td>-</td><td>Subtraction</td></tr><tr><td>xor</td><td>^</td><td>Bitwise exclusive or</td></tr></table>

These operations are atomic read-modify-write operations. For signed integer types, the arithmetic operation uses two's complement representation with silent wrap-around on overflow.

## 6.15.4.6 Atomic Modify Functions (64 Bits)

All OS: Metal 2.4 and later support the following atomic modify functions for Apple silicon. See the Metal Feature Set Tables to determine which GPUs support this feature.

These functions are applicable to any atomic object of type atomic_ulong. The only supported value for order is memory_order_relaxed.

void atomic_key_explicit(device A* object,

M operand,

memory_order order)

```cpp

void atomic_key_explicit(volatile device A* object,

    M operand,

    memory_order order)

```

The key in the function name is a placeholder for an operation name listed in the first column of Table 6.26, such as atomic_max_explicit. The operations detailed in Table 6.26 are arithmetic. The function atomically replaces the value pointed to by object with the result of the specified computation (third column of Table 6.26). The function returns void. There are no undefined results.

<div align="center">

Table 6.26. Atomic modify operations

</div>

<table border="1"><tr><td>Key</td><td>Operator</td><td>Computation</td></tr><tr><td>max</td><td>max</td><td>Compute max</td></tr><tr><td>min</td><td>min</td><td>Compute min</td></tr></table>

These operations are atomic read-modify-write operations.

## 6.16 Encoding Commands for Indirect Command Buffers

Indirect Command Buffers (ICBs) support the encoding of Metal commands into a Metal buffer for repeated use. Later, you can submit these encoded commands to the CPU or GPU for execution. ICBs for both render and compute commands use the command_buffer type to encode commands into an ICB object (represented in the Metal framework by MTLIndirectCommandBuffer):

```c

struct command_buffer {

    size_t size() const;

};

```

An ICB can contain either render or compute commands but not both. Execution of compute commands from a render encoder is illegal. So is execution of render commands from a compute encoder.

## 6.16.1 Encoding Render Commands in Indirect Command Buffers

All OS: Metal 2.1 and later support indirect command buffers for render commands.

ICBs allow the encoding of draw commands into a Metal buffer for subsequent execution on the GPU.

In a shading language function, use the command_buffer type to encode commands for ICBs into a Metal buffer object that provides indexed access to a render_command structure.

```c

struct arguments {

    command_buffer cmd_buffer;

};

kernel void producer(device arguments &args,

                    ushort cmd_idx [[thread_position_in_grid]])

{

    render_command cmd(args.cmd_buffer, cmd_idx);

    ...

}

```

render_command can encode any draw command type. The following public interface for render_command is defined in the header <metal_command_buffer>. To pass render_pipeline_state objects to your shader, use argument buffers. Within an argument buffer, the pipeline state can be passed as scalars or in an array.

set_render_pipeline_state(...) and render pipeline states are available in iOS in Metal 2.2 and later, and macOS in Metal 2.1 and later:

enum class primitive_type { point, line, line_strip, triangle, triangle_strip };

Metal 4 defines the following structures and enumerations:

```c

enum class cull_mode { none, front, back };

enum class depth_clip_mode { clip, clamp };

enum class triangle_fill_mode { fill, lines };

struct depth_stencil_state {

  public:

    depth_stencil_state();

    depth_stencil_state(const depth_stencil_state &);

    depth_stencil_state &operator=(const depth_stencil_state);

};

struct render_command {

public:

  explicit render_command(command_buffer icb, unsigned cmd_index);

  void set_render_pipeline_state(

      render_pipeline_state pipeline_state);

  template <typename T ...>

  void set_vertex_buffer(device T *buffer, uint index);

  template <typename T ...>

  void set_vertex_buffer(constant T *buffer, uint index);

  // Metal 3.1: Supported passing vertex strides.

  template <typename T ...>

  void set_vertex_buffer(device T *buffer, size_t stride,

                        uint index);

  template <typename T ...>

  void set_vertex_buffer(constant T *buffer, size_t stride,

                        uint index);

  // Metal 4: Support setting raster states.

  void set_cull_mode(cull_mode mode);

  void set_front_facing_winding(winding w);

  void set_triangle_fill_mode(triangle_fill_mode mode);

  // Metal 4: Set depth stencil states.

  void set_depth_bias(float bias, float slope_scale, float clamp);

  void set_depth_clip_mode(depth_clip_mode mode);

  void set_depth_stencil_state(depth_stencil_state state);

  template <typename T ...>

```

```cpp

void set_fragment_buffer(device T *buffer, uint index);

template <typename T ...>

void set_fragment_buffer(const T *buffer, uint index);

void draw_primitives(primitive_type type, uint vertex_start,

                  uint vertex_count, uint instance_count,

                  uint base_instance);

// Overloaded draw_indexed_primitives based on index_buffer.

void draw_indexed_primitives(primitive_type type,

                          uint index_count,

                          device ushort *index_buffer,

                          uint instance_count,

                          uint base_vertex,

                          uint base_instance);

void draw_indexed_primitives(primitive_type type,

                          uint index_count,

                          device uint *index_buffer,

                          uint instance_count,

                          uint base_vertex,

                          uint base_instance);

void draw_indexed_primitives(primitive_type type,

                          uint index_count,

                          constant ushort *index_buffer,

                          uint instance_count,

                          uint base_vertex,

                          uint base_instance);

void draw_indexed_primitives(primitive_type type,

                          uint index_count,

                          constant uint *index_buffer,

                          uint instance_count,

                          uint base_vertex,

                          uint base_instance);


// Overloaded draw_patches based on patch_index_buffer and

// tessellation_factor_buffer.

void draw_patches(uint number_of_patch_control_points,

                  uint patch_start, uint patch_count,

                  const device uint *patch_index_buffer,

                  uint instance_count, uint base_instance,

                  const device MTLQuadTessellationFactorsHalf

                      *tessellation_factor_buffer,

                  uint instance_stride = 0);

```

```cpp

void draw_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    const device uint *patch_index_buffer,

    uint instance_count, uint base_instance,

    const device

        MTLTriangleTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    const device uint *patch_index_buffer,

    uint instance_count, uint base_instance,

    constant MTLQuadTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    constant device uint *patch_index_buffer,

    uint instance_count, uint base_instance,

    constant MTLTriangleTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    constant device uint *patch_index_buffer,

    uint instance_count, uint base_instance,

    const device

        MTLTriangleTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    constant device uint *patch_index_buffer,

    uint instance_count, uint base_instance,

    constant MTLQuadTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

```

```cpp

void draw_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    constant uint *patch_index_buffer,

    uint instance_count, uint base_instance,

    constant MTLTriangleTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

```

// Overloaded draw_indexed_patches based on patch_index_buffer,

// control_point_index_buffer and tessellation_factor_buffer.

void draw_indexed_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    const device uint *patch_index_buffer,

    const device void *control_point_index_buffer,

    uint instance_count, uint base_instance,

    const device MTLQuadTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    const device uint *patch_index_buffer,

    const device void *control_point_index_buffer,

    uint instance_count, uint base_instance,

    const device MTLTriangleTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    const device uint *patch_index_buffer,

    const device void *control_point_index_buffer,

    uint instance_count, uint base_instance,

    constant MTLQuadTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    const device uint *patch_index_buffer,

    constant void *control_point_index_buffer,

    uint instance_count, uint base_instance,

    const device MTLQuadTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    const device uint *patch_index_buffer,

    constant void *control_point_index_buffer,

    uint instance_count, uint base_instance,

    const device MTLTriangleTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    const device uint *patch_index_buffer,

    constant void *control_point_index_buffer,

    uint instance_count, uint base_instance,

    constant MTLTriangleTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    constant uint *patch_index_buffer,

    constant device void *control_point_index_buffer,

    uint instance_count, uint base_instance,

    const device MTLQuadTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

constant uint *patch_index_buffer,

const device void *control_point_index_buffer,

uint instance_count, uint base_instance,

const device MTLTriangleTessellationFactorsHalf

    *tessellation_factor_buffer,

uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

uint patch_start, uint patch_count,

constant uint *patch_index_buffer,

const device void *control_point_index_buffer,

uint instance_count, uint base_instance,

constant MTLQuadTessellationFactorsHalf

    *tessellation_factor_buffer,

uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

uint patch_start, uint patch_count,

constant uint *patch_index_buffer,

const device void *control_point_index_buffer,

uint instance_count, uint base_instance,

constant MTLTriangleTessellationFactorsHalf

    *tessellation_factor_buffer,

uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

uint patch_start, uint patch_count,

constant uint *patch_index_buffer,

constant void *control_point_index_buffer,

uint instance_count, uint base_instance,

const device MTLTriangleTessellationFactorsHalf

    *tessellation_factor_buffer,

uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

uint patch_start, uint patch_count,

constant uint *patch_index_buffer,

constant void *control_point_index_buffer,

uint instance_count, uint base_instance,

const device MTLTriangleTessellationFactorsHalf

    *tessellation_factor_buffer,

uint instance_stride = 0);

```cpp

    constant MTLQuadTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

void draw_indexed_patches(uint number_of_patch_control_points,

    uint patch_start, uint patch_count,

    constant uint *patch_index_buffer,

    constant void *control_point_index_buffer,

    uint instance_count, uint base_instance,

    constant MTLTriangleTessellationFactorsHalf

        *tessellation_factor_buffer,

    uint instance_stride = 0);

// Reset the entire command. After reset(), without further

// modifications, execution of this command doesn't perform

// any action.

void reset();

// Copy the content of the `source` command into this command.

void copy_command(render_command source);

};

```

When accessing command_buffer, Metal does not check whether the access is within bounds. If an access is beyond the capacity of the buffer, the behavior is undefined.

- Calls to draw* methods in render_command encode the actions taken by the command. If multiple calls are made, only the last one takes effect.

The exposed methods in render_command mirror the interface of MTLIndirectRenderCommand and are similar to MTLRenderCommandEncoder. Notable differences with MTLRenderCommandEncoder are:

- The tessellation arguments are passed directly in render_command::draw_patches and render_command::draw_indexed_patches. Other calls do not set up the tessellation arguments.

## 6.16.2 Encoding Compute Commands in Indirect Command Buffers

iOS: Metal 2.2 and later support indirect command buffers for compute commands.

macOS: Metal 2.3 and later support indirect command buffers for compute commands.

ICBs allow the encoding of dispatch commands into a Metal buffer for subsequent execution on the GPU.

In a shading language function, use the command_buffer type to encode commands for ICBs into a Metal buffer object that provides indexed access to a compute_command structure:

```c

struct arguments {

    command_buffer cmd_buffer;

```

[kernel] void producer(device arguments &args,

    ushort cmd_idx [[thread_position_in_grid]])

{

    compute_command cmd(args.cmd_buffer, cmd_idx);

    ...

}

compute_command can encode any dispatch command type. The following public interface for compute_command is defined in the header <metal_command_buffer>. The compute_pipeline_state type represents compute pipeline states, which can only be passed to shaders through argument buffers. Within an argument buffer, the pipeline state can be passed as scalars or in an array.

struct compute_command {

public:

    explicit compute_command(command_buffer icb,

        unsigned cmd_index);

    void set_compute_pipeline_state(

        compute_pipeline_state pipeline);

    template <typename T ...>

    void set_kernel_buffer(device T *buffer, uint index);

    template <typename T ...>

    void set_kernel_buffer(constant T *buffer, uint index);

    // Metal 3.1: Supports passing kernel strides.

    template <typename T ...>

    void set_kernel_buffer(device T *buffer, size_t stride,

        uint index);

    template <typename T ...>

    void set_kernel_buffer(constant T *buffer, size_t stride,

        uint index);

    void set_barrier();

    void clear_barrier();

    void concurrent_dispatch_threadgroups(

        uint3 threadgroups_per_grid,

        uint3 threads_per_threadgroup);

    void concurrent_dispatch_threads(uint3 threads_per_grid,

        uint3 threads_per_threadgroup);

    void set_threadgroup_memory_length(uint length, uint index);

    void set_stage_in_region(uint3 origin, uint3 size);

    // Reset the entire command. After reset(), without further

Page 282 of 346

```cpp

// modifications. Execution of this command doesn't perform

// any action.

void reset();

// Copy the content of the `source` command into this command.

void copy_command(compute_command source);

};

```

When accessing command_buffer, Metal does not check whether the access is within bounds. If an access is beyond the capacity of the buffer, the behavior is undefined.

The exposed methods in compute_command mirror the interface of MTLIndirectComputeCommand and are similar to MTLComputeCommandEncoder.

In an ICB, dispatches are always concurrent. Calls to the concurrent_dispatch* methods in compute_command encode the actions taken by the command. If multiple calls are made only the last one takes effect.

The application is responsible for putting barriers where they are needed. Barriers encoded in an ICB do not affect the parent encoder.

The CPU may have initialized individual commands within a command_buffer before the command_buffer is passed as an argument to a shader. If the CPU has not already initialized a command, you must reset that command before using it.

## 6.16.3 Copying Commands of an Indirect Command Buffer

Copying a command structure (either render_command or compute_command) via operator= does not copy the content of the command, it only makes the destination command point to the same buffer and index as the source command. To copy the content of the command, call the copy_command functions listed in sections 6.16.1 and 6.16.2.

Copying is only supported between commands pointing to compatible command buffers. Two command buffers are compatible only if they have matching ICB descriptors (MTLIndirectCommandBufferDescriptor objects). The commands themselves must also refer to valid indexes within the buffers. The following example illustrates using copy_command to copy the content of a render command from cmd0 to cmd1:

```c

struct arguments {

    command_buffer cmd_buffer;

    render_pipeline_state pipeline_state_0;

    render_pipeline_state pipeline_state_1;

};

```

[[kernel]] void producer(device arguments &args) {

    render_command cmd0(args.cmd_buffer, 0);

    render_command cmd1(args.cmd_buffer, 1);

    cmd0.set_render_pipeline_state(args.pipeline_state_0);

    // Make the command at index 1 point to command at index 0.

    cmd1 = cmd0;

```

// Change the pipeline state for the command at index 0 in the

// buffer.

cmd1.set_render_pipeline_state(args.pipeline_state_0);

// The command at index 1 in the buffer is not yet modified.

cmd1 = render_command(args.cmd_buffer, 1);

// Copy the content of the command at index 0 to command at

// index 1.

cmd1.copy_command(cmd0);

## 6.17 Variable Rasterization Rate

iOS: Metal 2.2 and later support variable rasterization rate and the rasterization rate map.

macOS: Metal 2.3 and later support variable rasterization rate and the rasterization rate map.

Variable rasterization rate (VRR) can reduce the shading cost of high-resolution rendering by reducing the fragment shader invocation rate based on screen position. VRR is especially useful to avoid oversampling peripheral information in Augmented Reality (AR) / Virtual Reality (VR) applications.

rasterization_rate_map_decoder structure to describe the mapping of per-layer rasterization rate data. Each layer contains minimum quality values in screen space and can have a different physical fragment space dimension. For AR/VR, these quality values are based on the lens transform or eye-tracking information.

```c

struct rasterization_rate_map_data;

struct rasterization_rate_map_decoder {

    explicit rasterization_rate_map_decoder(

        constant rasterization_rate_map_data &data) thread;

    float2 map_screen_to_physical_coordinates(float2 screen_coordinates,

        uint layer_index = 0) const thread;

    uint2 map_screen_to_physical_coordinates(uint2 screen_coordinates,

        uint layer_index = 0) const thread;

    float2 map_physical_to_screen_coordinates(float2 physical_coordinates,

        uint layer_index = 0) const thread;

    uint2 map_physical_to_screen_coordinates(uint2 physical_coordinates,

        uint layer_index = 0) const thread;

};

```

The VRR map describes the mapping between screen space and physical fragment space and enables conversion of the rendering results back to the desired screen resolution. To convert between screen space and physical fragment space in the shader, the app must call the copyParameterDataToBuffer:offset: method of MTLRasterizationRateMap to fill the buffer with map data before using any of the conversion functions in the rasterization_rate_map_decoder structure. Passing anything other than a pointer to

the data exported by the copyParameterDataToBuffer:offset: method has an undefined behavior.

The following example shows how the app must pass the rasterization_rate_map_data at the shader bind point to the constructor of the rasterization_rate_map_decoder structure:

[[fragment]] float4 fragment_shader(/* other arguments */

    constant rasterization_rate_map_data &data [[buffer(0)]]) {

    float2 screen_coords = ...;

    rasterization_rate_map_decoder map(data);

    float2 physical_coords =

        map.map_screen_to_physical_coordinates(screen_coords);

    ...

}

Alternately, the app can compute the offset where the compiled data is stored and use an explicit cast or pointer arithmetic to form the data for a valid rasterization_rate_map_data. Since rasterization_rate_map_data is an incomplete type, some operations on it are inherently forbidden (such as pointer arithmetic on the pointer type or sizeof).

## 6.18 Ray-Tracing Functions

All OS: Metal 2.3 and later support ray-tracing functions.

Metal defines the ray-tracing functions and types in <metal_raytracing> in the namespace metal::raytracing. Metal 2.3 and later supports them only in a compute function (kernel function), except where noted below. Metal 2.4 and later offer additional support for them in vertex, fragment, and tile functions.

## 6.18.1 Acceleration Structure Functions

In Metal 2.3 and later, you can call one of the following functions to check if an acceleration structure (see section 2.17.7) is null:

bool

is_null_primitive_acceleration_structure(primitive_acceleration_structure)

bool

is_null_instance_acceleration_structure(instance_acceleration_structure)

In Metal 2.4 and later, you can call the following function to check if an acceleration structure is null:

bool

is_null_acceleration_structure(acceleration_structure<intersection_t

tags...>)

In Metal 3.1 and later, you can iterate over the acceleration structure referenced by an instance acceleration structure using the following functions:

- Call the following function to query the number of instances in an instance acceleration structure:

uint get_instance_count() const

- Call the following function to retrieve the acceleration structure referenced by an instance contained in an instance acceleration structure. The return type is the templatized type defined in section 2.17.7.

```cpp

template <typename... intersection_tags>

    acceleration_structure< intersection_tags...>

get_acceleration_structure(uint instance_id)

```

If the declared return type does not match the acceleration structure type reference by the instance contained in an instance acceleration structure, then the results are undefined. Instance acceleration structures that do not use instance and/or primitive motion tags can be returned as an acceleration structure type that does contain those tags. For example, an instance acceleration structure without any motion (instance or primitive) can be returned as:

- acceleration_structure<instancing>

- acceleration_structure<instancing, instance_motion>

- acceleration_structure<instancing, primitive_motion>

- acceleration_structure<instancing, primitive_motion, instance_motion>

This capability allows you to avoid providing a dedicated intersector for each set of tags when working with multiple acceleration structure types at the potential performance cost due to traversing an acceleration structure that does not require those tags.

## 6.18.2 Intersector Intersect Functions

After creating the intersector<intersection_tags...> object (see section 2.17.6), you can call one of the following intersect functions based on the value of the intersection_tags.

<div align="center">

Table 6.27. Intersect function

</div>

Function

result_type intersect(...parameters...).

<div align="center">

Table 6.28 shows the possible parameters for intersect function. All intersect functions must have ray and accel_struct parameter. The other parameters are optional.

</div>

<div align="center">

Table 6.28. Intersect functions input parameters

</div>

<table border="1"><tr><td>Parameter</td><td>Description</td></tr><tr><td>ray</td><td>Ray properties</td></tr><tr><td>accel_struct</td><td>Acceleration structure of type acceleration_structure intersection_tags...</td></tr><tr><td>mask</td><td>Intersection mask to be AND'd with instance mask defined in the Metal API MTLAccelerationStructureInstanceDescriptor. Instances with nonoverlapping masks will be skipped.</td></tr><tr><td>time</td><td>The time associated with the ray. The parameter exists if the intersection_tags have primitive_motion or instance_motion.</td></tr><tr><td>func_table</td><td>Intersection function table of type intersection_function_table intersection_tags...</td></tr><tr><td>payload</td><td>User payload object, which is passed by reference. When the user calls intersect(), the payload parameter is copied to the ray_data address space and passed to the intersection function. The result is copied on the exit of the intersection function (section 5.1.6) and the payload object is updated.</td></tr><tr><td>ifba</td><td>If the intersection_tags include intersection_function_buffer, you may optionally pass an object of type intersection_function_buffer_arguments (see section 6.18.8). The ifba.intersection_function_buffer must be uniform within the SIMD-group of the call.</td></tr><tr><td>user_data</td><td>If the intersection_tags include user_data, you may optionally pass a buffer pointing to user data for the intersection function. If you pass a buffer, you also need to pass ifba.</td></tr><tr><td>All OS: Metal 4 and later.</td><td></td></tr></table>

The result_type is

using result_type = intersection_result<intersection_tags...>;

The following set of intersect functions are available only if intersection_tags does not have instancing:

```cpp

result_type

intersect(

    ray ray,

    primitive_acceleration_structure accel_struct) const;

result_type

intersect(

    ray ray,

    primitive_acceleration_structure accel_struct,

    intersection_function_table<intersection_tags...> func_table)

const;

template <typename T>

result_type

intersect(

    ray ray,

    primitive_acceleration_structure accel_struct,

    intersection_function_table<intersection_tags...> func_table,

    thread T &payload) const;

```

The following set of intersect functions are available only if intersection_tags has instancing:

```cpp

result_type

intersect(

    ray ray,

    instance_acceleration_structure accel_struct,

    uint mask = ~0U) const;

result_type

intersect(

    ray ray,

    instance_acceleration_structure accel_struct,

    intersection_function_table<intersection_tags...> func_table)

const;

```

The following set of intersect functions are available only if intersection_tags has instancing and don't have an intersection_function_buffer:

```cpp

template <typename T>

    result_type

    intersect(

        ray ray,

        instance_acceleration_structure accel_struct,

        intersection_function_table<intersection_tags...> func_table,

        thread T &payload) const;

result_type

intersect(

    ray ray,

    instance_acceleration_structure accel_struct,

    uint mask,

    intersection_function_table<intersection_tags...> func_table)

const;

template <typename T>

    result_type

    intersect(

        ray ray,

        instance_acceleration_structure accel_struct,

        uint mask,

        intersection_function_table<intersection_tags...> func_table,

        thread T &payload) const;

```

In Metal 2.4 and later, the following set of intersect functions are available if intersection_tags have primitive_motion or instance_motion:

```cpp

template <typename T, intersection_tags...>

    result_type

    intersect(

        ray ray,

        acceleration_structure< intersection_tags...> accel_struct,

        float time) const;

```

The following set of intersect functions are available only if intersection_tags has instancing and don't have an intersection_function_buffer:

template <typename T, intersection_tags...>

result_type

intersect(

  ray ray,

  acceleration_structure< intersection_tags...> accel_struct,

  float time,

  intersection_function_table<intersection_tags...> func_table)

const;

template <typename T, intersection_tags...>

result_type

intersect(

  ray ray,

  acceleration_structure< intersection_tags...> accel_struct,

  float time,

  intersection_function_table<intersection_tags...> func_table,

  thread T &payload) const;

In Metal 2.4 and later, the following set of intersect functions are available only if intersection_tags have instancing and either primitive_motion or instance_motion:

template <typename T, intersection_tags...>

result_type

intersect(

  ray ray,

  acceleration_structure< intersection_tags...> accel_struct,

  uint mask = ~0U,

  float time = 0.0f) const;

The following set of intersect functions are available only if intersection_tags has instancing, and either primitive_motion or instance_motion don’t have an intersection_function_buffer:

template <typename T, intersection_tags...>

result_type

intersect(

  ray ray,

  acceleration_structure< intersection_tags...> accel_struct,

  uint mask,

  float time,

  intersection_function_table<intersection_tags...> func_table)

const;

template <typename T, intersection_tags...>

Page 290 of 346

result_type

intersect(

    ray ray,

    acceleration_structure< intersection_tags...> accel_struct,

    uint mask,

    float time,

    intersection_function_table< intersection_tags...> func_table,

    thread T &payload) const;

In Metal 3.2 and later, it's possible to avoid a copy and directly access the memory of the intersection by using intersection_result_ref<intersection_tags...> (see section 2.17.5) and the ray_data payload pointer in a callback:

```cpp

template <typename Callable>

void intersect(..., Callable callback)

template <typename Payload, typename Callable>

void intersect(..., const thread Payload &payload_in,

                Callable callback)

```

The lifetime is the intersection_result_ref and the ray_data payload pointer is the duration of the callback. If you store the intersection_result_ref or payload pointer and use it after the intersect() call completes, the behavior is undefined because the system may free the memory. You can't perform recursive ray tracing within the callback body. After the callback exits, the shader is free to intersect rays again.

The following is an example of the use of a lambda with the intersection_result_ref:

```cpp

[[kernel]] void trace_rays_with_payload(...) {

    intersector<instancing, max_levels<2>, triangle_data> i;

    i.intersect(ray, acceleration_structure, MyPayload{},

        [&](intersection_result_ref<instancing, max_levels<2>,

triangle_data> result,

            const ray_data MyPayload &final_payload)

    {

        result.get_primitive_id();

        // ...

    });

}

```

In Metal 4 and later, the following set of intersect functions are available only if intersection_tags has an intersection_function_buffer and doesn't have instancing:

result_type

intersect(

```cpp

ray ray,

acceleration_structure<> accel_struct,

intersection_function_buffer_ifba) const;

template <typename T>

result_type

intersect(

    ray ray,

    acceleration_structure<> accel_struct,

    intersection_function_buffer_ifba,

    thread T &payload) const;

```

In Metal 4 and later, the following set of intersect functions are available only if intersection_tags has an intersection_function_buffer and instancing:

```cpp

result_type

intersect(

    ray ray,

    acceleration_structure<instancing> accel_struct,

    intersection_function_buffer_ifba) const;

template <typename T>

result_type

intersect(

    ray ray,

    acceleration_structure<instancing> accel_struct,

    intersection_function_buffer_ifba,

    thread T &payload) const;

result_type

intersect(

    ray ray,

    uint mask,

    acceleration_structure<instancing> accel_struct,

    intersection_function_buffer_ifba) const;

template <typename T>

result_type

intersect(

    ray ray,

    uint mask,

    acceleration_structure<instancing> accel_struct,

    intersection_function_buffer_ifba,

    thread T &payload) const;

```

In Metal 4 and later, the following set of intersect functions are available only if

intersection_tags has an intersection_function_buffer, instancing, and

primitive_motion.

```cpp

result_type

intersect(

    ray ray,

    acceleration_structure<instancing, primitive_motion> as,

    float time,

    intersection_function_buffer_ifba) const;

template <typename T>

result_type

intersect(

    ray ray,

    acceleration_structure<instancing, primitive_motion> as,

    float time,

    intersection_function_buffer_ifba,

    thread T &payload) const;

result_type

intersect(

    ray ray,

    uint mask,

    float time,

    acceleration_structure<instancing, primitive_motion> as,

    intersection_function_buffer_ifba) const;

template <typename T>

result_type

intersect(

    ray ray,

    uint mask,

    float time,

    acceleration_structure<instancing, primitive_motion> as,

    intersection_function_buffer_ifba,

    thread T &payload) const;

```

In Metal 4 and later, the following set of intersect functions are available only if

intersection_tags has an intersection_function_buffer, instancing, and

instance_motion:

```cpp

result_type

intersect(

```

ray ray,

acceleration_structure<instancing, instance_motion> as,

float time,

intersection_function_buffer_ifba) const;

template <typename T>

result_type

intersect(

    ray ray,

    acceleration_structure<instancing, instance_motion> as,

    float time,

    intersection_function_buffer_ifba,

    thread T &payload) const;

result_type

intersect(

    ray ray,

    uint mask,

    float time,

    acceleration_structure<instancing, instance_motion> as,

    intersection_function_buffer_ifba) const;

template <typename T>

result_type

intersect(

    ray ray,

    uint mask,

    float time,

    acceleration_structure<instancing, instance_motion> as,

    intersection_function_buffer_ifba,

    thread T &payload) const;

In Metal 4 and later, the following set of intersect functions are available only if intersection_tags has an intersection_function_buffer, user_data, and doesn't have instancing:

```cpp

result_type

intersect(

    ray ray,

    acceleration_structure<> accel_struct,

    intersection_function_buffer_ifba,

    const device void *user_data) const;

template <typename T>

```

result_type

intersect(

  ray ray,

  acceleration_structure<> accel_struct,

  intersection_function_buffer_ifba,

  const device void *user_data,

  thread T &payload) const;

In Metal 4 and later, the following set of intersect functions are available only if

intersection_tags has an intersection_function_buffer, user_data, and

instancing:

result_type

intersect(

  ray ray,

  acceleration_structure<instancing> accel_struct,

  intersection_function_buffer_ifba,

  const device void *user_data) const;

template <typename T>

result_type

intersect(

  ray ray,

  acceleration_structure<instancing> accel_struct,

  intersection_function_buffer_ifba,

  const device void *user_data,

  thread T &payload) const;

result_type

intersect(

  ray ray,

  uint mask,

  acceleration_structure<instancing> accel_struct,

  intersection_function_buffer_ifba,

  const device void *user_data) const;

template <typename T>

result_type

intersect(

  ray ray,

  uint mask,

  acceleration_structure<instancing> accel_struct,

  intersection_function_buffer_ifba,

  const device void *user_data,

  thread T &payload) const;

Page 295 of 346

In Metal 4 and later, the following set of intersect functions are available only if intersection_tags has an intersection_function_buffer, user_data, instancing, and primitive_motion:

result_type

intersect(

  ray ray,

  acceleration_structure<instancing, primitive_motion> as,

  float time,

  intersection_function_buffer_ifba,

  const device void *user_data) const;

template <typename T>

result_type

intersect(

  ray ray,

  acceleration_structure<instancing, primitive_motion> as,

  float time,

  intersection_function_buffer_ifba,

  const device void *user_data,

  thread T &payload) const;

result_type

intersect(

  ray ray,

  uint mask,

  float time,

  acceleration_structure<instancing, primitive_motion> as,

  intersection_function_buffer_ifba,

  const device void *user_data) const;

template <typename T>

result_type

intersect(

  ray ray,

  uint mask,

  float time,

  acceleration_structure<instancing, primitive_motion> as,

  intersection_function_buffer_ifba,

  const device void *user_data,

  thread T &payload) const;

In Metal 4 and later, the following set of intersect functions are available only if intersection_tags has an intersection_function_buffer, instancing, user_data, and instance_motion:

ser_data, and instance_motion:

result_type

intersect(

    ray ray,

    acceleration_structure<instancing, instance_motion> as,

    float time,

    intersection_function_buffer_ifba,

    const device void *user_data) const;

template <typename T>

result_type

intersect(

    ray ray,

    acceleration_structure<instancing, instance_motion> as,

    float time,

    intersection_function_buffer_ifba,

    const device void *user_data,

    thread T &payload) const;

result_type

intersect(

    ray ray,

    uint mask,

    float time,

    acceleration_structure<instancing, instance_motion> as,

    intersection_function_buffer_ifba,

    const device void *user_data) const;

template <typename T>

result_type

intersect(

    ray ray,

    uint mask,

    float time,

    acceleration_structure<instancing, instance_motion> as,

    intersection_function_buffer_ifba,

    const device void *user_data,

    thread T &payload) const;

## 6.18.3 Intersector Functions to Control Traversal Behavior

All OS: Metal 3.1 adds support for curves.

To override the default behavior of the traversal, you can use the following member functions of intersector<intersection_tags...> object.

<div align="center">

Table 6.29. Intersect functions to control traversal

</div>

<table border="1"><tr><td>Functions to control traversal behavior</td></tr><tr><td>void set_triangle_front_facing_winding(winding)</td></tr><tr><td>void set_geometry_cull_mode(geometry_cull_mode)</td></tr><tr><td>void set_opacity_cull_mode(opacity_cull_mode)</td></tr><tr><td>void force_opacity(forced_opacity)</td></tr><tr><td>void assume_geometry_type(geometry_type)</td></tr><tr><td>void assume_identity_transforms(bool)</td></tr><tr><td>void accept_any_intersection(bool)</td></tr></table>

Triangles have two sides or "faces". The front facing winding determines which triangle face is considered the "front" face when viewed from the ray origin. If the vertices appear in clockwise order when viewed from the ray origin and the front facing winding is clockwise, then the visible face is the front face. The other face is the back face. If the front facing winding is counterclockwise, then the opposite is true. Use the following function to change the default winding (clockwise):

enum class winding {

    clockwise,

    counterclockwise

};

void set_triangle_front_facing_winding(winding w);

To change the default triangle cull mode (none), use the following function:

```cpp

enum class triangle_cull_mode {

    none,

    front,

    back

};

```

void set_triangle_cull_mode(triangle_cull_mode tcm);

If the cull mode is set to front, then triangles whose front face is visible from the ray origin are not considered for intersection. Otherwise, if the cull mode is set to back, then triangles whose back face is visible from the ray origin are not considered for intersection.

The following function may be used to set the intersector to cull all bounding box or triangle primitives from the set of candidate geometries. The default geometry cull mode is none.

```cpp

enum class geometry_cull_mode {

    none,

    triangle,

    bounding_box,

    curve          // Metal 3.1 and later.

};

void set_geometry_cull_mode(geometry_cull_mode gcm);

```

The default opacity cull mode is none. Use the following function to change the opacity. See below on how opacity affects triangle and bounding box primitives.

```cpp

enum class opacity_cull_mode {

    none,

    opaque,

    non_opaque

};

void set_opacity_cull_mode(opacity_cull_mode ocm);

```

Call the following function to override per-instance and per-geometry setting of forced capacity. The default is none.

```cpp

enum class forced_opacity {

    none,

    opaque,

    non_opaque

};

void force_opacity(forced_opacity fo);

```

Triangle primitives may also be culled based on their opacity: An opaque triangle will not run any intersection function. A non_opaque triangle runs its intersection function to accept or reject the hit.

The PrimitiveAccelerationStructure encodes if the triangle is opaque or non_opaque by declaring MTLAccelerationStructureGeometryFlagOpaque. The opaqueness can be overridden by calling intersector.force_opacity().If used, this

takes precedence over the per-instance opaqueness flags (MTLaccelerationStructureInstanceFlagOpaque and MTLaccelerationStructureInstanceFlagNonOpaque), which in turn takes precedence over the per-geometry opaqueness.

For custom bounding box primitives, the opaqueness will be evaluated in the same way as described for triangles (first intersector.set_opacity_cull_mode(), then InstanceFlags, then GeometryFlags). The opaque parameter informs the bounding box intersection program the resolved opaqueness state. The intersection function may then use this to influence its evaluation of if a hit is encountered or not.

intersector.set_opacity_cull_mode() skips over primitive types based on their opaqueness.

If intersector.force_opacity() is set to opaque or non_opaque, then intersector.set_opacity_cull_mode() must be none. The reverse is also true: Opacity Override and Opacity culling cannot be mixed. The results of illegal combinations are undefined.

Use the following functions to declare if the acceleration structure contains a triangle, bounding box, and/or curve geometry. The default geometry is geometry_type::triangle geometry_type::bounding_box. By default, Metal assumes acceleration structure will not contain curve geometry to improve performance. Call assume_geometry_type with a value that includes geometry_type::curve to enable curves to be intersected in an intersect call or intersection query step.

```cpp

enum class geometry_type {

    none,

    triangle,

    bounding_box,

    curve, // Metal 3.1 and later.

    all

};

void assume_geometry_type(geometry_type gt)

```

To set the intersector object to assume identify transforms, call the following function with the value true. The default is false.

void assume_identity_transforms(bool value);

To set the intersector object to immediately return the first intersection it finds, call the following function with the value true. The default is false. One use of this function is when you only need to know if one point is visible from another, such as when rendering shadows or ambient occlusion.

void accept_any_intersection(bool value);

In Metal 3.1 and later, use the following functions to add hints to the intersector and intersection_query to specify the curve basis, the number of control points, and the curve type to optimize traversal for specific curve types:

Note that curve_basis is an enumerated type and not a bitmask.

```cpp

enum class curve_basis {

    bspline,

    catmull_rom,

    linear,

    bezier,

    all,

};

enum class curve_type {

    round,

    flat,

    all,

};

```

Use the following function to set the curve basis function to assume. Defaults to curve_basis::all, meaning that all curve basis functions will be enabled.

void assume_curve_basis(curve_basis cb)

Use the following function to set the curve type to assume. Defaults to curve_type::all, meaning that both curve types will be enabled.

void assume_curve_type(curve_type ct)

Use the following function to set the number of curve control points to assume. Defaults to 0, meaning that any number of control points, as appropriate for the assumed curve basis (if any), will be enabled. Other valid options are 2, 3, or 4, depending on the curve basis.

void assume_curve_control_point_count(uint n)

## 6.18.4 Intersector Functions for Ray Contribution and Geometry Multiplier

All OS: Metal 4 adds support to specify Ray Contribution and Geometry Multiplier.

In Metal 4 and later, you can specify the ray contribution and geometry multiplier by adding state per intersector object if if intersection_tags has intersection_function_buffer. Note the calculation of base index and geometry multiplier use the lower 4 bits.

Call the following function to set the base ID. The default value of the base ID is 0.

void set_base_id(uint index);

Call the following function to set the geometry multiplier on the intersector. The default value of multiplier is 1.

void set_geometry_multiplier(uint multiplier);

## 6.18.5 Intersection Query Functions

All OS: Metal 2.4 and later support intersection query functions.

All OS: Metal 3.1 and later support intersection query functions for curves.

To start traversals and query traversal specific information, create an intersection query object (see section 2.17.8) with a nondefault constructor or first call reset(...). If not called in this sequence, the behavior is undefined.

Table 6.30, Table 6.32, and Table 6.33 show the list of functions that can be called depending on the geometry type encountered during the traversal, assuming next() has returned true. Note that some functions come in pairs: a candidate and a committed primitive. When next() is called for the first time, the primitive reported after the traversal is always a candidate until the user commits the primitive by calling commit_triangle_intersection(), commit_bounding_box_intersection(), or commit_curve_intersection() on the query object. Note that opaque triangles, tested without user intersection, commit automatically when intersected.

<div align="center">

Table 6.30. Intersection query functions

</div>

<table border="1"><tr><td>Functions</td><td>Triangle</td><td>Bounding</td><td>Curve</td></tr><tr><td>void reset(...</td><td>√</td><td>√</td><td>√</td></tr><tr><td>bool next()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>void abort()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>intersection_typeget_candidate_intersection_type()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>intersection_typeget_committed_intersection_type()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>void commit_triangle_intersection()</td><td>√</td><td></td><td></td></tr><tr><td>voidcommit_bounding_box_intersection(float distance)</td><td></td><td>√</td><td></td></tr><tr><td>voidcommit_curve_intersection()
All OS: Metal 3.1 and later.</td><td></td><td></td><td>√</td></tr></table>

<div align="center">

Table 6.31. Intersection query functions with max_levels<Count>

</div>

<table border="1"><tr><td>Functions</td><td>Triangle</td><td>Bounding</td><td>Curve</td></tr><tr><td>uint get_candidate_instance_count()
All OS: Metal 3.1 and later.</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_candidate_instance_id(uint depth)
All OS: Metal 3.1 and later.</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_candidate_user_instance_id(uint depth)
All OS: Metal 3.1 and later.</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_committed_instance_count()
All OS: Metal 3.1 and later.</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_committed_instance_id(uint depth)
All OS: Metal 3.1 and later.</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_committed_user_instance_id(uint depth)
All OS: Metal 3.1 and later.</td><td>√</td><td>√</td><td>√</td></tr></table>

<div align="center">

Table 6.32. Intersection query ray value functions

</div>

<table border="1"><tr><td>Ray values functions</td><td>Triangle</td><td>Bounding</td><td>Curve</td></tr><tr><td>float3 get_world_space_ray_origin()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float3 get_world_space_ray_direction()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float get_ray_min_distance()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>intersection_params get_intersection_params()</td><td>√</td><td>√</td><td>√</td></tr></table>

<div align="center">

Table 6.33. Intersection query candidate value functions

</div>

<table border="1"><tr><td>Candidate intersections value functions</td><td>Triangle</td><td>Bounding</td><td>Curve</td></tr><tr><td>float get_candidate_triangle_distance()</td><td>√</td><td></td><td></td></tr><tr><td>uint get_candidate_instance_id()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_candidate_user_instance_id()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_candidate_geometry_id()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_candidate_primitive_id()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float2 get_candidate_triangle_barycentric_coord()</td><td>√</td><td></td><td></td></tr><tr><td>bool is_candidate_non_opaque_bounding_box()</td><td></td><td>√</td><td></td></tr><tr><td>bool is_candidate_triangle_front_facing()</td><td>√</td><td></td><td></td></tr><tr><td>float4x3 get_candidate_object_to_world_transform()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float4x3 get_candidate_world_to_object_transform()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float3 get_candidate_ray_origin()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float3 get_candidate_ray_direction()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>const device void * get_candidate_primitive_data()</td><td>√</td><td>√</td><td>√</td></tr><tr><td colspan="4">All OS: Metal 3 and later.</td></tr></table>

<div align="center">

Table 6.34. Intersect query committed value functions

</div>

<table border="1"><tr><td>Committed intersections value functions</td><td>Triangle</td><td>Bounding</td><td>Curve</td></tr><tr><td>float get_committed_distance()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_committed_instance_id()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_committed_user_instance_id()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_committed_geometry_id()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>uint get_committed_primitive_id()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float2 get_committed_triangle_barycentric_coord()</td><td>√</td><td></td><td></td></tr><tr><td>bool is_committed_triangle_front_facing()</td><td>√</td><td></td><td></td></tr><tr><td>float4x3 get_committed_object_to_world_transform()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float4x3 get_committed_world_to_object_transform()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float3 get_committed_ray_origin()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>float3 get_committed_ray_direction()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>const device void * get_committed_primitive_data()</td><td>√</td><td>√</td><td>√</td></tr><tr><td>All OS: Metal 3 and later.</td><td></td><td></td><td></td></tr><tr><td>float get_candidate_curve_parameter()</td><td></td><td></td><td>√</td></tr><tr><td>All OS: Metal 3.1 and later.</td><td></td><td></td><td>√</td></tr><tr><td>float get_committed_curve_parameter()</td><td></td><td></td><td>√</td></tr><tr><td>All OS: Metal 3.1 and later.</td><td></td><td></td><td>√</td></tr></table>

In Metal 3.1 and later, intersection query supports the following functions when specified with the max_levels<Count> intersection tags:

- Call the following function to query the distance of a candidate triangle hit that needs consideration:

float get_candidate_triangle_distance();

- Call the following function to query the distance of the currently committed hit:

float get_committed_distance();

- Call the following function to query the top-level structure instance ID for the current candidate hit:

uint get_candidate_instance_id();

- Call the following function to query user instance ID provided by user on the bottom level acceleration structure for the current candidate hit:

```c

uint get_candidate_user_instance_id();

```

uint get_candidate_geometry_id();

- Call the following function to query the bottom-level structure geometry ID for the current candidate hit:

- Call the following function to query the bottom-level structure primitive ID within the geometry for the current candidate hit:

```c

uint get_candidate_primitive_id();

```

- Call the following function to query the top-level structure instance ID for the current committed hit:

```c

uint get_committed_instance_id();

```

- Call the following function to query user instance ID provided by user on the bottom level acceleration structure for the current committed hit:

```c

uint get_committed_user_instance_id();

```

- Call the following function to query the bottom-level structure geometry ID for the current committed hit:

uint get_committed_geometry_id();

- Call the following function to query the bottom-level structure primitive ID within the geometry for the current committed hit:

uint get_committed_primitive_id();

- Call the following function to query the ray origin in object space for the current hit candidate:

float3 get_candidate_ray_origin();

float3 get_candidate_ray_direction();

- Call the following function to query the ray direction in object space for the current hit candidate:

float3 get_committed_ray_origin();

- Call the following function to query the ray origin in object space for the current committed hit:

- Call the following function to query the ray direction in object space for the current committed hit:

float3 get_committed_ray_direction();

- Call the following function to query the matrix for transforming ray origin/direction of current hit candidate from object-space to world-space:

float4x3 get_candidate_object_to_world_transform();

- Call the following function to query the matrix for transforming ray origin/direction of current candidate hit from world-space to object-space:

float4x3 get_candidate_world_to_object_transform();

- Call the following function to query the matrix for transforming ray origin/direction of current committed hit from object-space to world-space:

float4x3 get_committed_object_to_world_transform();

- Call the following function to query the matrix for transforming ray origin/direction of current committed hit from world-space to object-space:

float4x3 get_committed_world_to_object_transform();

- Call the following function to query the candidate hit location barycentric coordinates. Valid when get_candidate_intersection_type() returns triangle:

float2 get_candidate_triangle_barycentric_coord();

- For vertex attributes v0, v1, and v2, the value at the specified barycentric point is:

v1 * barycentric_coord.x +

v2 * barycentric_coord.y +

v0 * (1.0f - (barycentric_coord.x + barycentric_coord.y))

- Call the following function to query the committed hit location barycentric coordinates. Valid when get_committed_intersection_type() returns triangle: float2 get_committed_triangle_barycentric_coord();

- Call the following function to query if the hit triangle candidate is front or back facing. Returns true if it is front face and false if it is back face. Valid when get_candidate_intersection_type() returns triangle:

bool is_candidate_triangle_front_facing();

- Call the following function to query if the committed hit is front or back facing. Returns true if it is front face and false if it is back face. Valid when get_committed_intersection_type() returns triangle:

bool is_committed_triangle_front_facing();

- Call the following function to query the per-primitive data for the current candidate primitive:

```cpp

const device void *get_candidate_primitive_data();

```

- Call the following function to query the per-primitive data for the current committed hit:

```javascript

const device void *get_committed_primitive_data();

```

In Metal 3.1 and later, the following two functions can be called when get_candidate_intersection_type() returns curve and the intersection tag has curve_data:

- Call the following to query the curve parameter for the current candidate curve: float get_candidate_curve_parameter();

- Call the following to query the curve parameter for the current committed intersection. Valid when get_candidate_intersection_type() returns curve. float get_committed_curve_parameter();

In Metal 3.1 and later, the rest of the functions in this section can be called when the intersection tag has max_levels<Count>:

- Call the following function to query the number of instances in the candidate intersection:

uint get_candidate_instance_count();

- Call the following function to query the instance ID at level depth in the candidate intersection.

uint get_candidate_instance_id(uint depth);

- Call the following function to query the user instance ID at level depth in the candidate intersection:

uint get_candidate_user_instance_id(uint depth);

- Call the following function to query the number of instances in the committed intersection:

uint get_committed_instance_count();

- Call the following function to query the instance ID at level depth in the committed intersection:

  uint get_committed_instance_id(uint depth);

- Call the following function to query the user instance ID at level depth in the committed intersection:

  uint get_committed_user_instance_id(uint depth);

## 6.18.6 Indirect Instance Descriptors

In Metal 3.1 and later, you can fill out indirect instance descriptors from the GPU. Metal provides the following type definitions:

```c

enum MTLAccelerationStructureInstanceOptions : uint

{

    MTLAccelerationStructureInstanceOptionNone = 0,

    MTLAccelerationStructureInstanceOptionDisableTriangleCulling =

        (1 << 0),

MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise = (1 << 1),

    MTLAccelerationStructureInstanceOptionOpaque = (1 << 2),

    MTLAccelerationStructureInstanceOptionNonOpaque = (1 << 3),

};

typedef packed_float3 MTLPackedFloat3;

typedef packed_float3 MTLPackedFloat4x3[4];

struct MTLAccelerationStructureInstanceDescriptor

{

    MTLPackedFloat4x3 transformationMatrix;

    MTLAccelerationStructureInstanceOptions options;

    uint mask;

    uint intersectionFunctionTableOffset;

    uint accelerationStructureIndex;

};

struct MTLAccelerationStructureUserIDInstanceDescriptor

{

    MTLPackedFloat4x3 transformationMatrix;

    MTLAccelerationStructureInstanceOptions options;

    uint mask;

    uint intersectionFunctionTableOffset;

    uint accelerationStructureIndex;

    uint userID;

};

```

To facilitate filing out the descriptor, Metal provides an implicit conversion from acceleration_structure<intersection_tags...> to MTLResourceID. acceleration_structure<primitive_motion> primitiveAStruct = ...; MTLResourceID resource_id = primitiveAStruct;

## 6.18.7 Curve Utility Functions

Metal 3.1 and later provide a set of curve utility functions that Metal defines in the header <metal_curves>. It uses the following abbreviations:

Ps is float or half.

P is a scalar or a vector of Ps. If Ps is float, P is float4.

The functions return the position or the first or second derivative on a curve given a curve parameter t, and control points p0, p1, etc. As shown in Table 6.35, the functions support quadratic Bezier, cubic Bezier, quadratic B-Spline, cubic B-Spline, cubic Hermite, and CatmullRom curves.

<div align="center">

Table 6.35. Curve utility functions

</div>

<table border="1"><tr><td>Function</td><td>Description</td></tr><tr><td>P bezier(Ps_t,P p0,P p1,P p2)</td><td>Returns the position on a quadratic Bézier curve</td></tr><tr><td>P bezier_derivative(Ps_t,P p0,P p1,P p2)</td><td>Returns the first derivative on a quadratic Bézier curve</td></tr><tr><td>P bezier_second_derivative(Ps_t,P p0,P p1,P p2)</td><td>Returns the second derivative on a quadratic Bézier curve</td></tr><tr><td>P bezier(Ps_t,P p0,P p1,P p2,P p3)</td><td>Returns the position on a cubic Bézier curve</td></tr><tr><td>P bezier_derivative(Ps_t,P p0,P p1,P p2,P p3)</td><td>Returns the first derivative on a cubic Bézier curve</td></tr><tr><td>P bezier_second_derivative(Ps_t,P p0,P p1,P p2,P p3)</td><td>Returns the second derivative on a cubic Bézier curve</td></tr><tr><td>P bspline(Ps_t,P p0,P p1,P p2)</td><td>Returns the position on a quadratic B-spline curve</td></tr><tr><td>P bspline_derivative(Ps_t,P p0,P p1,P p2)</td><td>Returns the first derivative on a quadratic B-spline curve</td></tr><tr><td>P bspline_second_derivative(Ps_t,P p0,P p1,P p2)</td><td>Returns the second derivative on a quadratic B-spline curve</td></tr><tr><td>P bspline(Ps_t,P p0,P p1,P p2,P p3)</td><td>Returns the position on a cubic B-spline curve</td></tr></table>

<table border="1"><tr><td>Function</td><td>Description</td></tr><tr><td>P bspline_derivative(Ps_t,Pp0,Pp1,Pp2,Pp3)</td><td>Returns the first derivative on a cubic B-spline curve</td></tr><tr><td>P bspline_second_derivative(Ps_t,Pp0,Pp1,Pp2,Pp3)</td><td>Returns the second derivative on a cubic B-spline curve</td></tr><tr><td>P hermite(Ps_t,Pp0,Pp1,Pm0,Pm1)</td><td>Returns the position on a cubic Hermite curve</td></tr><tr><td>P hermite_derivative(Ps_t,Pp0,Pp1,Pm0,Pm1)</td><td>Returns the first derivative on a cubic Hermite curve</td></tr><tr><td>P hermite_second_derivative(Ps_t,Pp0,Pp1,Pm0,Pm1)</td><td>Returns the second derivative on a cubic Hermite curve</td></tr><tr><td>P catmull_rom(Ps_t,Pp0,Pp1,Pp2,Pp3)</td><td>Returns the position on a Catmull-Rom curve</td></tr><tr><td>P catmull_rom_derivative(Ps_t,Pp0,Pp1,Pp2,Pp3)</td><td>Returns the first derivative on a Catmull-Rom curve</td></tr><tr><td>P catmull_rom_second_derivative(Ps_t,Pp0,Pp1,Pp2,Pp3)</td><td>Returns the second derivative on a Catmull-Rom curve</td></tr></table>

## 6.18.8 Intersection Function Buffer Descriptors

In Metal 4 and later, you can use indirect function buffers to associate geometry in a scene with a set of shaders that operate on that geometry in the acceleration structure. The user provides a buffer containing intersection_function_buffer_arguments.

```c

struct intersection_function_buffer_arguments

{

    // Buffer containing instruction function handles aligned

    // to 8 bytes.

    const device void * intersection_function_buffer;

    // Maximum range in bytes

    size_t intersection_function_buffer_size;

    // The stride between intersection function entries.

    size_t intersection_function_stride;

};

```

The stride, intersection_function_stride, support ranges from [0.4096] in 8 bytes increments.

For convenience, the header provides the Metal MTLIntersectionFunctionBufferArguments which is convertible to intersection_function_buffer_arguments.

The example above passes a buffer to intersect (see section 6.18.2).

## 6.19 Logging Functions

All OS: Metal 3.2 and later support logging for Apple silicon.

Metal defines the logging functions and types in <metal_logging>. To enable logging, you need to set -fmetal-enable-logging (see section 1.6.9).

```c

enum log_type

{

    log_type_debug,    // Captures verbose information useful only for

                    // debugging your code.

    log_type_info,    // Captures information that is helpful to

                    // troubleshoot problems.

    log_type_default, // Captures information that is essential for

                    // troubleshooting problems.

    log_type_error,    // Captures errors that occur during the

                    // execution of your code.

    log_type_fault     // Captures information about faults and bugs

                    // in your code.

}

```

```c

struct os_log

{

    os_log(constant char *subsystem, constant char *category)

constant;

    void log_with_type(log_type type, constant char *format, ...)

constant;

    void log_debug(constant char *format, ...) constant;

    void log_info(constant char *format, ...) constant;

    void log(constant char *format, ...) constant;

    void log_error(constant char *format, ...) constant;

    void log_fault(constant char *format, ...) constant;

};

```

The os_log logging methods support most of the format specifiers that std::printf supports in C++, with the following exceptions:

- They don't support the %n and %s conversion specifiers.

- They don't support the %@ and %. *P and custom format specifiers that the CPU os_log supports.

- Metal supports the h1 length modifier for 4-byte types like int and float, which you need to use when printing vectors.

- Vectors may print with %v[num_elements][length_modifier][conversion_specifier]. For example, a float4 can print with %v4h1f while a uchar2 can print as %v2hhu.

- Default argument promotion applies to arguments of half type which promote to the double type. Default argument promotion doesn't apply to vectors.

- The format string must be a string literal.

Shaders can perform logging by defining an os_log object and using any of the log member functions:

```cpp

constant metal::os_log custom_log("com.custom_log.subsystem",

    "custom category");

void test_log(float x) {

    if (x < M_PI_F)

        custom_log.log("custom message %f", x);

}

A default os_log object os_log_default is available to use instead of a custom os_log object:

void test_log(float x) {

    if (x < M_PI_F)

        os_log_default.log("custom message %f", x);

}

```

Metal places messages from the shader into a log buffer with a size that MTLLogState determines. All the draw/dispatches in a command buffer share the log buffer. The system only removes the messages from the log buffer when the command buffer completes. Because multiple command buffers can share a log buffer, the system may block the removal of the messages until other command buffers complete. When the log buffer becomes full, the system drops all subsequent messages. Logging resumes after the CPU has an opportunity to empty the log buffer.

By default, messages that the CPU reads from the log buffer go into the unified logging system with the corresponding subsystem, category, and level. Messages that os_log_default logs go into the CPU unified logging system with the corresponding level and subsystem/category being nil. For custom handling of shader logging messages, see the Metal API's addLogHandler.
