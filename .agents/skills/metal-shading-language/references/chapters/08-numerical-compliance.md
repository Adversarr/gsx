## 8 Numerical Compliance

This chapter covers how Metal represents floating-point numbers regarding accuracy in mathematical operations. Metal is compliant to a subset of the IEEE 754 standard.

## 8.1 INF, NaN, and Denormalized Numbers

INF must be supported for single-precision, half-precision, and bfloat floating-point numbers.

NaNs must be supported for single-precision, half-precision, and bfloat floating-point numbers (with fast math disabled). If fast math is enabled the behavior of handling NaN or INF (as inputs or outputs) is undefined. Signaling NaNs are not supported.

Denormalized single-precision, half-precision, or bfloat floating-point numbers passed as input to or produced as the output of single-precision, half-precision, or bfloat floating-point arithmetic operations may be flushed to zero.

## 8.2 Rounding Mode

Either round ties to even or round toward zero rounding mode may be supported for single-precision, half-precision, and bfloat floating-point operations.

## 8.3 Floating-Point Exceptions

Floating-point exceptions are disabled in Metal.

## 8.4 ULPs and Relative Error

Table 8.1 describes the minimum accuracy of single-precision floating-point basic arithmetic operations and math functions given as ULP values. The reference value used to compute the ULP value of an arithmetic operation is the infinitely precise result.

<div align="center">

Table 8.1. Accuracy of single-precision floating-point operations and functions

</div>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>x+y</td><td>Correctly rounded</td></tr><tr><td>x-y</td><td>Correctly rounded</td></tr><tr><td>x*y</td><td>Correctly rounded</td></tr><tr><td>1.0/x</td><td>Correctly rounded</td></tr><tr><td>x/y</td><td>Correctly rounded</td></tr></table>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>acos</td><td>&lt;= 4 ulp</td></tr><tr><td>acosh</td><td>&lt;= 4 ulp</td></tr><tr><td>asin</td><td>&lt;= 4 ulp</td></tr><tr><td>asinh</td><td>&lt;= 4 ulp</td></tr><tr><td>atan</td><td>&lt;= 5 ulp</td></tr><tr><td>atan2</td><td>&lt;= 6 ulp</td></tr><tr><td>atanh</td><td>&lt;= 5 ulp</td></tr><tr><td>ceil</td><td>Correctly rounded</td></tr><tr><td>copysign</td><td>0 ulp</td></tr><tr><td>cos</td><td>&lt;= 4 ulp</td></tr><tr><td>cosh</td><td>&lt;= 4 ulp</td></tr><tr><td>cospi</td><td>&lt;= 4 ulp</td></tr><tr><td>exp</td><td>&lt;= 4 ulp</td></tr><tr><td>exp2</td><td>&lt;= 4 ulp</td></tr><tr><td>exp10</td><td>&lt;= 4 ulp</td></tr><tr><td>fabs</td><td>0 ulp</td></tr><tr><td>fdim</td><td>Correctly rounded</td></tr><tr><td>floor</td><td>Correctly rounded</td></tr><tr><td>fma</td><td>Correctly rounded</td></tr><tr><td>fmax</td><td>0 ulp</td></tr><tr><td>fmin</td><td>0 ulp</td></tr><tr><td>fmod</td><td>0 ulp</td></tr><tr><td>fract</td><td>Correctly rounded</td></tr><tr><td>frexp</td><td>0 ulp</td></tr><tr><td>ilogb</td><td>0 ulp</td></tr><tr><td>ldexp</td><td>Correctly rounded</td></tr><tr><td>log</td><td>&lt;= 4 ulp</td></tr><tr><td>log2</td><td>&lt;= 4 ulp</td></tr></table>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>log10</td><td>&lt;= 4 ulp</td></tr><tr><td>modf</td><td>0 ulp</td></tr><tr><td>nextafter</td><td>0 ulp</td></tr><tr><td>pow</td><td>&lt;= 16 ulp</td></tr><tr><td>powr</td><td>&lt;= 16 ulp</td></tr><tr><td>rint</td><td>Correctly rounded</td></tr><tr><td>round</td><td>Correctly rounded</td></tr><tr><td>rsqrt</td><td>Correctly rounded</td></tr><tr><td>sin</td><td>&lt;= 4 ulp</td></tr><tr><td>sincos</td><td>&lt;= 4 ulp</td></tr><tr><td>sinh</td><td>&lt;= 4 ulp</td></tr><tr><td>sinpi</td><td>&lt;= 4 ulp</td></tr><tr><td>sqrt</td><td>Correctly rounded</td></tr><tr><td>tan</td><td>&lt;= 6 ulp</td></tr><tr><td>tanpi</td><td>&lt;= 6 ulp</td></tr><tr><td>tanh</td><td>&lt;= 5 ulp</td></tr><tr><td>trunc</td><td>Correctly rounded</td></tr></table>

Table 8.2 describes the minimum accuracy of single-precision floating-point arithmetic operations given as ULP values with fast math enabled (which is the default unless you specify -fno-fast-math as a compiler option).

<div align="center">

Table 8.2. Accuracy of single-precision operations and functions with fast math enabled

</div>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>x+y</td><td>Correctly rounded</td></tr><tr><td>x-y</td><td>Correctly rounded</td></tr><tr><td>x*y</td><td>Correctly rounded</td></tr><tr><td>1.0/x</td><td>&lt;= 1 ulp for x in the domain of $ 2^{-126} $ to $ 2^{126} $</td></tr></table>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>x/y</td><td>&lt;= 2.5 ulp for y in the domain of $ 2^{-126} $ to $ 2^{126} $</td></tr><tr><td>acos(x)</td><td>&lt;= 5 ulp for x in the domain [-1, 1]</td></tr><tr><td>acosh(x)</td><td>Implemented as log(x + sqrt(x*x - 1.0))</td></tr><tr><td>asin(x)</td><td>&lt;= 5 ulp for x in the domain [-1, 1] and |x| &gt;= $ 2^{-125} $</td></tr><tr><td>asinh(x)</td><td>Implemented as log(x + sqrt(x*x + 1.0))</td></tr><tr><td>atan(x)</td><td>&lt;= 5 ulp</td></tr><tr><td>atanh(x)</td><td>Implemented as 0.5*(log(1.0 + x)/(1.0 - x))</td></tr><tr><td>atan2(y,x)</td><td>Implemented as: if x &gt; 0, atan(y/x); if x &lt; 0 and y &gt; 0, atan(y/x) + M_PI_F; if x &lt; 0 and y &lt; 0, atan(y/x) - M_PI_F; and if x = 0 or y = 0, the result is undefined.</td></tr><tr><td>ceil</td><td>Correctly rounded</td></tr><tr><td>copysign</td><td>0 ulp</td></tr><tr><td>cos(x)</td><td>For x in the domain [-pi, pi], the maximum absolute error is &lt;= $ 2^{-13} $ and larger otherwise.</td></tr><tr><td>cosh(x)</td><td>Implemented as 0.5*(exp(x) + exp(-x))</td></tr><tr><td>cospi(x)</td><td>For x in the domain [-1, 1], the maximum absolute error is &lt;= $ 2^{-13} $ and larger otherwise.</td></tr><tr><td>exp(x)</td><td>&lt;=3 + floor(fabs(2*x)) ulp</td></tr><tr><td>exp2(x)</td><td>&lt;=3 + floor(fabs(2*x)) ulp</td></tr><tr><td>exp10(x)</td><td>Implemented as exp2(x*log2(10))</td></tr><tr><td>fabs</td><td>0 ulp</td></tr><tr><td>fdim</td><td>Correctly rounded</td></tr><tr><td>floor</td><td>Correctly rounded</td></tr><tr><td>fma</td><td>Correctly rounded</td></tr><tr><td>fmax</td><td>0 ulp</td></tr><tr><td>fmin</td><td>0 ulp</td></tr><tr><td>fmod</td><td>Undefined</td></tr></table>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>fract</td><td>Correctly rounded</td></tr><tr><td>frexp</td><td>0 ulp</td></tr><tr><td>ilogb</td><td>0 ulp</td></tr><tr><td>ldexp</td><td>Correctly rounded</td></tr><tr><td>log(x)</td><td>For x in the domain [0.5, 2], the maximum absolute error is &lt;= $ 2^{-21} $; otherwise if x &gt; 0 the maximum error is &lt;=3 ulp; otherwise the results are undefined.</td></tr><tr><td>log2(x)</td><td>For x in the domain [0.5, 2], the maximum absolute error is &lt;= $ 2^{-22} $; otherwise if x &gt; 0 the maximum error is &lt;=2 ulp; otherwise the results are undefined.</td></tr><tr><td>log10(x)</td><td>Implemented as log2(x)*log10(2)</td></tr><tr><td>modf</td><td>0 ulp</td></tr><tr><td>pow(x,y)</td><td>Implemented as exp2(y*log2(x)). Undefined for x = 0 and y = 0.</td></tr><tr><td>powr(x,y)</td><td>Implemented as exp2(y*log2(x)). Undefined for x = 0 and y = 0.</td></tr><tr><td>rint</td><td>Correctly rounded</td></tr><tr><td>round(x)</td><td>Correctly rounded</td></tr><tr><td>rsqrt</td><td>&lt;=2 ulp</td></tr><tr><td>sin(x)</td><td>For x in the domain [-pi, pi], the maximum absolute error is &lt;= $ 2^{-13} $ and larger otherwise.</td></tr><tr><td>sinh(x)</td><td>Implemented as 0.5*(exp(x) - exp(-x))</td></tr><tr><td>sincos(x)</td><td>ULP values as defined for sin(x) and cos(x)</td></tr><tr><td>sinpi(x)</td><td>For x in the domain [-1, 1], the maximum absolute error is &lt;= $ 2^{-13} $ and larger otherwise.</td></tr><tr><td>sqrt(x)</td><td>Implemented as x*rsqrt(x) with special cases handled correctly.</td></tr><tr><td>tan(x)</td><td>Implemented as sin(x)*(1.0/cos(x))</td></tr><tr><td>tanh(x)</td><td>Implemented as (t - 1.0)/(t + 1.0), where t = exp(2.0*x)</td></tr><tr><td>tanpi(x)</td><td>Implemented as tan(x*pi)</td></tr><tr><td>trunc</td><td>Correctly rounded</td></tr></table>

Table 8.3 describes the minimum accuracy of half-precision floating-point basic arithmetic operations and math functions given as ULP values. Table 8.3 applies to iOS and macOS, starting with Apple GPU Family 4 hardware.

<div align="center">

Table 8.3. Accuracy of half-precision floating-point operations and functions

</div>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>x+y</td><td>Correctly rounded</td></tr><tr><td>x-y</td><td>Correctly rounded</td></tr><tr><td>x*y</td><td>Correctly rounded</td></tr><tr><td>1.0/x</td><td>Correctly rounded</td></tr><tr><td>x/y</td><td>Correctly rounded</td></tr><tr><td>acos(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>acosh(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>asin(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>asinh(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>atan(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>atanh(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>atan2(y,x)</td><td>&lt;= 1 ulp</td></tr><tr><td>ceil</td><td>Correctly rounded</td></tr><tr><td>copysign</td><td>0 ulp</td></tr><tr><td>cos(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>cosh(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>cospi(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>exp(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>exp2(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>exp10(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>fabs</td><td>0 ulp</td></tr><tr><td>fdim</td><td>Correctly rounded</td></tr><tr><td>floor</td><td>Correctly rounded</td></tr></table>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>fma</td><td>Correctly rounded</td></tr><tr><td>fmax</td><td>0 ulp</td></tr><tr><td>fmin</td><td>0 ulp</td></tr><tr><td>fmod</td><td>0 ulp</td></tr><tr><td>fract</td><td>Correctly rounded</td></tr><tr><td>frexp</td><td>0 ulp</td></tr><tr><td>ilogb</td><td>0 ulp</td></tr><tr><td>ldexp</td><td>Correctly rounded</td></tr><tr><td>log(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>log2(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>log10(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>modf</td><td>0 ulp</td></tr><tr><td>nextafter</td><td>0 ulp</td></tr><tr><td>rint</td><td>Correctly rounded</td></tr><tr><td>round(x)</td><td>Correctly rounded</td></tr><tr><td>rsqrt</td><td>Correctly rounded</td></tr><tr><td>sin(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>sinh(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>sincos(x)</td><td>ULP values as defined for sin(x) and cos(x)</td></tr><tr><td>sinpi(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>sqrt(x)</td><td>Correctly rounded</td></tr><tr><td>tan(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>tanh(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>tanpi(x)</td><td>&lt;= 1 ulp</td></tr><tr><td>trunc</td><td>Correctly rounded</td></tr></table>

<div align="center">

Table 8.4 describes the minimum accuracy of bfloat floating-point basic arithmetic operations and math functions given as ULP values. Table 8.4 applies to all OS, starting with Apple GPU Family 6 or Metal GPU Family 3.

</div>

<div align="center">

Table 8.4. Accuracy of bfloat floating-point operations and functions

</div>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>x+y</td><td>Correctly rounded</td></tr><tr><td>x-y</td><td>Correctly rounded</td></tr><tr><td>x*y</td><td>Correctly rounded</td></tr><tr><td>1.0/x</td><td>Correctly rounded</td></tr><tr><td>x/y</td><td>Correctly rounded</td></tr></table>

<div align="center">

Table 8.5. Accuracy of bfloat floating-point operations and functions with fast math enabled

</div>

<table border="1"><tr><td>Math function</td><td>Minimum accuracy(ULP values)</td></tr><tr><td>x+y</td><td>Correctly rounded</td></tr><tr><td>x-y</td><td>Correctly rounded</td></tr><tr><td>x*y</td><td>Correctly rounded</td></tr><tr><td>1.0/x</td><td>&lt;=0.6ulp for x in the domain of $ 2^{-126} $ to $ 2^{126} $</td></tr><tr><td>x/y</td><td>&lt;=0.6ulp for y in the domain of $ 2^{-126} $ to $ 2^{126} $</td></tr></table>

Even though the precision of individual math operations and functions are specified in Table 8.1, Table 8.2, Table 8.3, Table 8.4, and Table 8.5, the Metal compiler, in fast math mode (see section 1.6.5), may do various optimizations like reassociate floating-point operations that may dramatically change floating-point results. Reassociation may change or ignore the sign of zero, allow optimizations to assume the arguments and result are not NaN or +/-INF, inhibit or create underflow or overflow, and thus cannot be used in code that relies on rounding behavior such as $ \left(x+2^{52}\right)-2^{52} $, or ordered floating-point comparisons.

The ULP is defined as follows:

If x is a real number that lies between two finite consecutive floating-point numbers a and b, without being equal to one of them, then $ \mathrm{ulp}(x)=|b-a| $, otherwise $ \mathrm{ulp}(x) $ is the distance between the two nonequal finite floating-point numbers nearest x. Moreover, $ \mathrm{ulp}(\mathrm{NaN}) $ is NaN.

## 8.5 Edge Case Behavior in Flush to Zero Mode

If denormalized values are flushed to zero, then a function may return one of four results:

1. Any conforming result when not in flush to zero mode.

2. If the result given by step 1 is a subnormal before rounding, it may be flushed to zero.

3. Any nonflushed conforming result for the function if one or more of its subnormal operands are flushed to zero.

4. If the result of step 3 is a subnormal before rounding, the result may be flushed to zero. In each of the above cases, if an operand or result is flushed to zero, the sign of the zero is undefined.

## 8.6 Conversion Rules for Floating-Point and Integer Types

When converting from a floating-point type to an integer, the conversion uses round toward zero rounding mode. Use the "round ties to even" or "round toward zero" rounding mode for conversions from a floating-point or integer type to a floating-point type.

The conversions from half and bfloat to float are lossless. Conversions from float to half or to bfloat round the mantissa using the round ties to even rounding mode. When converting a float to a half, denormalized numbers generated for the half data type may not be flushed to zero.

When converting a floating-point type to an integer type, if the floating-point value is NaN, the resulting integer is 0.

Note that fast math does not change the accuracy of conversion operations.

## 8.7 Texture Addressing and Conversion Rules

The texture coordinates specified to the sample, sample_compare, gather, gather_compare, read, and write functions cannot be INF or NaN. An out-of-bound texture read returns the default value for each component, as described in section 6.12, and Metal ignores an out-of-bound texture write.

The following sections discuss the application of conversion rules when reading and writing textures in a graphics or kernel function. When performing a multisample resolve operation, these conversion rules do not apply.

## 8.7.1 Conversion Rules for Normalized Integer Pixel Data Types

This section discusses converting normalized integer pixel data types to floating-point values and vice-versa.

## 8.7.1.1 Converting Normalized Integer Pixel Data Types to Floating-Point Values

For textures that have 8-,10-, or 16-bit normalized unsigned integer pixel values, the texture sample and read functions convert the pixel values from an 8- or 16-bit unsigned integer to a normalized single- or half-precision floating-point value in the range [0.0 ... 1.0].

For textures that have 8- or 16-bit normalized signed integer pixel values, the texture sample and read functions convert the pixel values from an 8- or 16-bit signed integer to a normalized single- or half-precision floating-point value in the range $ [-1.0\dots 1.0] $.

These conversions are performed as listed in the second column of Table 8.6. The precision of the conversion rules is guaranteed to be <= 1.5 ulp, except for the cases described in the "Corner Cases" column.

<div align="center">

Table 8.6. Conversion to a normalized float value

</div>

<table border="1"><tr><td>Convert from</td><td>Conversion rule to normalized float</td><td>Corner cases</td></tr><tr><td>1-bit normalized unsigned integer</td><td>float(c)</td><td>0 must convert to 0.0; 1 must convert to 1.0</td></tr><tr><td>2-bit normalized unsigned integer</td><td>float(c)/3.0</td><td>0 must convert to 0.0; 3 must convert to 1.0</td></tr><tr><td>4-bit normalized unsigned integer</td><td>float(c)/15.0</td><td>0 must convert to 0.0; 15 must convert to 1.0</td></tr><tr><td>5-bit normalized unsigned integer</td><td>float(c)/31.0</td><td>0 must convert to 0.0; 31 must convert to 1.0</td></tr><tr><td>6-bit normalized unsigned integer</td><td>float(c)/63.0</td><td>0 must convert to 0.0; 63 must convert to 1.0</td></tr><tr><td>8-bit normalized unsigned integer</td><td>float(c)/255.0</td><td>0 must convert to 0.0; 255 must convert to 1.0</td></tr><tr><td>10-bit normalized unsigned integer</td><td>float(c)/1023.0</td><td>0 must convert to 0.0; 1023 must convert to 1.0</td></tr><tr><td>16-bit normalized unsigned integer</td><td>float(c)/65535.0</td><td>0 must convert to 0.0; 65535 must convert to 1.0</td></tr><tr><td>8-bit normalized signed integer</td><td>max(-1.0, float(c)/127.0)</td><td>-128 and -127 must convert to -1.0; 0 must convert to 0.0; 127 must convert to 1.0</td></tr><tr><td>16-bit normalized signed integer</td><td>max(-1.0, float(c)/32767.0)</td><td>-32768 and -32767 must convert to -1.0; 0 must convert to 0.0; 32767 must convert to 1.0</td></tr></table>

## 8.7.1.2 Converting Floating-Point Values to Normalized Integer Pixel Data Types

For textures that have 8-, 10-, or 16-bit normalized unsigned integer pixel values, the texture write functions convert the single- or half-precision floating-point pixel value to an 8- or 16-bit unsigned integer.

For textures that have 8- or 16-bit normalized signed integer pixel values, the texture write functions convert the single- or half-precision floating-point pixel value to an 8- or 16-bit signed integer.

NaN values are converted to zero.

Conversions from floating-point values to normalized integer values are performed as listed in Table 8.7.

<div align="center">

Table 8.7. Conversion from floating-point to a normalized integer value

</div>

<table border="1"><tr><td>Convert to</td><td>Conversion rule to normalized integer</td></tr><tr><td>1-bit normalized unsigned integer</td><td>x = min(max(f, 0.0), 1.0); i0:0 = intRTNE(x)</td></tr><tr><td>2-bit normalized unsigned integer</td><td>x = min(max(f*3.0, 0.0), 3.0); i1:0 = intRTNE(x)</td></tr><tr><td>4-bit normalized unsigned integer</td><td>x = min(max(f*15.0, 0.0), 15.0); i3:0 = intRTNE(x)</td></tr><tr><td>5-bit normalized unsigned integer</td><td>x = min(max(f*31.0, 0.0), 31.0); i4:0 = intRTNE(x)</td></tr><tr><td>6-bit normalized unsigned integer</td><td>x = min(max(f*63.0, 0.0), 63.0); i5:0 = intRTNE(x)</td></tr><tr><td>8-bit normalized unsigned integer</td><td>x = min(max(f*255.0, 0.0), 255.0); i7:0 = intRTNE(x)</td></tr><tr><td>10-bit normalized unsigned integer</td><td>x = min(max(f*1023.0, 0.0), 1023.0); i9:0 = intRTNE(x)</td></tr><tr><td>16-bit normalized unsigned integer</td><td>x = min(max(f*65535.0, 0.0), 65535.0); i15:0 = intRTNE(x)</td></tr><tr><td>8-bit normalized signed integer</td><td>x = min(max(f*127.0, -127.0), 127.0); i7:0 = intRTNE(x)</td></tr><tr><td>16-bit normalized signed integer</td><td>x = min(max(f*32767.0, -32767.0), 32767.0); i15:0 = intRTNE(x)</td></tr></table>

In Metal 2, all conversions to and from unorm data types round correctly.

## 8.7.2 Conversion Rules for Half-Precision Floating-Point Pixel Data Type

For textures that have half-precision floating-point pixel color values, the conversions from half to float are lossless. Conversions from float to half round the mantissa using the round ties to even rounding mode. Denormalized numbers for the half data type which may be generated when converting a float to a half may not be flushed to zero. A float NaN may

be converted to an appropriate NaN or be flushed to zero in the half type. A float INF must be converted to an appropriate INF in the half type.

## 8.7.3 Conversion Rules for Single-Precision Floating-Point Pixel Data Type

The following rules apply for reading and writing textures that have single-precision floating-point pixel color values:

- NaNs may be converted to a NaN value(s) or be flushed to zero.

- INFs must be preserved.

- Denormalized numbers may be flushed to zero.

- All other values must be preserved.

## 8.7.4 Conversion Rules for 10- and 11-bit Floating-Point Pixel Data Type

The floating-point formats use 5 bits for the exponent, with 5 bits of mantissa for 10-bit floating-point types, or 6 bits of mantissa for 11-bit floating-point types with an additional hidden bit for both types. There is no sign bit. The 10- and 11-bit floating-point types preserve denormalized values.

These floating-point formats use the following rules:

- If the exponent and mantissa are 0, the floating-point value is 0.0.

- If the exponent is 31 and the mantissa is != 0, the resulting floating-point value is a NaN.

- If the exponent is 31 and the mantissa is 0, the resulting floating-point value is positive infinity.

- If 1 <= exponent <= 30, the floating-point value is $ 2^{(\mathrm{exponent} - 15)} * (1 + \mathrm{mantissa}/N) $.

- If the exponent is 0 and the mantissa is != 0, the floating-point value is a denormalized number given as $ 2^{-14} * (\mathrm{mantissa}/N) $. If mantissa is 5 bits, N is 32; if mantissa is 6 bits, N is 64.

Conversion of a 10- or 11-bit floating-point pixel data type to a half- or single-precision floating-point value is lossless. Conversion of a half or single precision floating-point value to a 10- or 11-bit floating-point value must be <= 0.5 ULP. Any operation that results in a value less than zero for these floating-point types is clamped to zero.

## 8.7.5 Conversion Rules for 9-bit Floating-Point Pixel Data Type with a 5-bit Exponent

The RGB9E5_SharedExponent shared exponent floating-point format uses 5 bits for the exponent and 9 bits for the mantissa. There is no sign bit.

Conversion from this format to a half- or single-precision floating-point value is lossless and computed as $ 2^{(\mathrm{sharedExponent}-15)} * (\mathrm{mantissa}/512) $ for each color channel.

Conversion from a half or single precision floating-point RGB color value to this format is performed as follows, where N is the number of mantissa bits per component (9), B is the exponent bias (15) and Emax is the maximum allowed biased exponent value (31).

- Clamp the r, g, and b components (in the process, mapping NaN to zero) as follows:

rc = max(0, min(sharedexpmax, r))

gc = max(0, min(sharedexpmax, g))

bc = max(0, min(sharedexpmax, b))

Where $ \mathrm{sharedexpmax}=((2^N-1)/2^N) * 2^{(\mathrm{Emax}-\mathrm{B})} $.

- Determine the largest clamped component maxc:

maxc = max(rc, gc, bc)

- Compute a preliminary shared exponent expp:

expp = max(-B - 1, floor(log2(maxc)) + 1 + B)

- Compute a refined shared exponent exps:

$$
\mathrm{maxs} = \operatorname{floor}\left(\max c / 2^{(\mathrm{expp} - \mathrm{B} - \mathrm{N})} + 0.5f\right)
$$

$ \mathrm{exps}=\mathrm{expp} $, if $ 0 <= \mathrm{maxs} < 2^N $, and $ \mathrm{exps}=\mathrm{expp}+1 $, if $ \mathrm{maxs}=2^N $. 

- Finally, compute three integer values in the range 0 to 2N-1:

$$
\mathrm{rs} = \operatorname{floor}(\mathrm{rc} / 2^{(\mathrm{exps} - \mathrm{B} - \mathrm{N})} + 0.5f)
$$

$$
\mathrm{gs} = \operatorname{floor}(\mathrm{gc} / 2^{(\mathrm{exps} - \mathrm{B} - \mathrm{N})} + 0.5f)
$$

$$
\mathrm{bs} = \operatorname{floor}(\mathrm{bc} / 2^{(\mathrm{exps} - \mathrm{B} - \mathrm{N})} + 0.5f)
$$

Conversion of a half- or single-precision floating-point color values to the

MTLPixelFormatRGB9E5Float shared exponent floating-point value is <= 0.5 ULP.

## 8.7.6 Conversion Rules for Signed and Unsigned Integer Pixel Data Types

For textures that have an 8- or 16-bit signed or unsigned integer pixel values, the texture sample and read functions return a signed or unsigned 32-bit integer pixel value. The conversions described in this section must be correctly saturated.

Writes to these integer textures perform one of the conversions listed in Table 8.8.

<div align="center">

Table 8.8. Conversion between integer pixel data types

</div>

<table border="1"><tr><td>Convert from</td><td>To</td><td>Conversion rule</td></tr><tr><td>32-bit signed integer</td><td>8-bit signed integer</td><td>result = convert_char_saturate(val)</td></tr><tr><td>32-bit signed integer</td><td>16-bit signed integer</td><td>result = convert_short_saturate(val)</td></tr><tr><td>32-bit unsigned integer</td><td>8-bit unsigned integer</td><td>result = convert_uchar_saturate(val)</td></tr><tr><td>32-bit unsigned integer</td><td>16-bit unsigned integer</td><td>result = convert_ushort_saturate(val)</td></tr></table>

## 8.7.7 Conversion Rules for sRGBA and sBGRA Textures

Conversion from sRGB space to linear space is automatically done when sampling from an sRGB texture. The conversion from sRGB to linear RGB is performed before the filter specified in the sampler specified when sampling the texture is applied. If the texture has an alpha channel, the alpha data is stored in linear color space.

Conversion from linear to sRGB space is automatically done when writing to an sRGB texture. If the texture has an alpha channel, the alpha data is stored in linear color space.

The following is the conversion rule for converting a normalized 8-bit unsigned integer from an sRGB color value to a floating-point linear RGB color value (call it c):

```cpp

if (c <= 0.04045)

    result = c / 12.92;

else

    result = powr((c + 0.055) / 1.055, 2.4);

```

The precision of the above conversion must ensure that the delta between the resulting infinitely precise floating-point value when converting result back to an unnormalized sRGB value but without rounding to an 8-bit unsigned integer value (call it r) and the original sRGB 8-bit unsigned integer color value (call it rorig) is <= 0.5; for example:

fabs(r - rorig) <= 0.5

Use the following rules for converting a linear RGB floating-point color value (call it c) to a normalized 8-bit unsigned integer sRGB value:

if (isnan(c)) c = 0.0;

if (c > 1.0)

    c = 1.0;

else if (c < 0.0)

    c = 0.0;

else if (c < 0.0031308)

    c = 12.92 * c;

else

    c = 1.055 * powr(c, 1.0/2.4) - 0.055;

// Convert to integer scale: c = c * 255.0.

// Convert to integer: c = c + 0.5.

// Drop the decimal fraction.

// Convert the remaining floating-point(integral) value

// to an integer.

The precision of the above conversion shall be:

fabs(reference result - integer result) < 1.0.
