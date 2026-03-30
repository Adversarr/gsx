## 3 Operators

All OS: Metal 1 and later support scalar, vector, and matrix operators.

For indirect command buffers, the assignment operator (=) does not copy the contents of a command. For more about copying commands in indirect command buffers, see section 6.16.3.

## 3.1 Scalar and Vector Operators

This section lists both binary and unary operators and describes their actions on scalar and vector operands.

1. The arithmetic binary operators, add (+), subtract (-), multiply (*) and divide (/), act upon scalar and vector, integer, and floating-point data type operands. Following the usual arithmetic conversions, all arithmetic operators return a result of the same built-in type (integer or floating-point) as the type of the operands. After conversion, the following cases are valid:

- If the two operands of the arithmetic binary operator are scalars, the result of the operation is a scalar.

- If one operand is a scalar, and the other operand is a vector,

- The scalar converts to the element type that the vector operand uses.

- The scalar type then widens to a vector that has the same number of components as the vector operand.

- Metal performs the operation componentwise, which results in a same size vector.

- If the two operands are vectors of the same size, Metal performs the operation componentwise, which results in a same size vector.

Division on integer types that result in a value that lies outside of the range bounded by the maximum and minimum representable values of the integer type, such as TYPE_MIN/-1 for signed integer types or division by zero, does not cause an exception but results in an unspecified value. Division by zero for floating-point types results in $ \pm\infty $ or NaN, as prescribed by IEEE-754. (For more about the numerical accuracy of floating-point operations, see section 8.)

Because bfloat and half are not implicitly convertible to each other, the operators do not support mixing bfloat and half.

2. The modulus operator (%) acts upon scalar and vector integer data type operands. The modulus operator returns a result of the same built-in type as the type of the operands, after the usual arithmetic conversions. The following cases are valid:

- If the two operands of the modulus operator are scalars, the result of the operation is a scalar.

- If one operand is a scalar, and the other is a vector:

- The scalar converts to the element type of the vector operand.

- The scalar type then widens to a vector that has the same number of components as the vector operand.

- Metal performs the operation componentwise, which results in a same-size vector.

- If the two operands are vectors of the same size, Metal performs the operation componentwise, which results in a same-size vector.

For any component computed with a second operand that is zero, the modulus operator result is undefined. If one or both operands are negative, the results are undefined. Results for other components with nonzero operands remain defined.

If both operands are nonnegative, the remainder is nonnegative.

3. The arithmetic unary operators (+ and -) act upon scalar and vector, integer, and floating-point type operands.

4. The arithmetic post- and pre-increment and decrement operators (++ and --) have scalar and vector integer type operands. All unary operators work componentwise on their operands. The result is the same type as the operand. For post- and pre-increment and decrement, the expression needs to be assignable to an lvalue. Pre-increment and predecrement add or subtract 1 to the contents of the expression on which they operate, and the value of the pre-increment or predecrement expression is the resulting value of that modification. Post-increment and post-decrement expressions add or subtract 1 to the contents of the expression on which they operate, but the resulting expression has the expression's value before execution of the post-increment or post-decrement.

5. The relational operators [greater-than (>), less-than (<), greater-than or equal to (>=), and less-than or equal to (<=)] act upon scalar and vector, integer, and floating-point type operands. The result is a Boolean (bool type) scalar or vector. After converting the operand type, the following cases are valid:

- If the two operands of the relational operator are scalars, the result of the operation is a Boolean.

- If one operand is a scalar, and the other is a vector:

- The scalar converts to the element type of the vector operand.

- The scalar type then widens to a vector that has the same number of components as the vector operand.

- Metal performs the operation componentwise, which results in a Boolean vector.

- If the two operands are vectors of the same size, Metal performs the operation componentwise, which results in a same-size Boolean vector.

If either argument is a NaN, the relational operator returns false. To test a relational operation on any or all elements of a vector, use the any and all built-in functions in the context of an if(...) statement. (For more about any and all functions, see section 6.4.)

6. The equality operators, equal (==) and not equal (!=), act upon scalar and vector, integer and floating-point type operands. All equality operators result in a Boolean scalar or vector. After converting the operand type, the following cases are valid:

- If the two operands of the equality operator are scalars, the result of the operation is a Boolean.

- If one operand is a scalar, and the other is a vector:

- The scalar converts to the element type of the vector operand.

- The scalar type then widens to a vector that has the same number of components as the vector operand.

- Metal performs the operation componentwise, which results in a Boolean vector.

- If the two operands are vectors of the same size, Metal performs the operation componentwise, which results in a same-size Boolean vector.

All other cases of implicit conversions are illegal. If one or both arguments is NaN, the equality operator equal (==) returns false. If one or both arguments is NaN, the equality operator not equal (!=) returns true.

7. The bitwise operators [and (&), or (|), exclusive or (^), not (~)] can act upon all scalar and vector built-in type operands, except the built-in scalar and vector floating-point types.

- For built-in vector types, Metal applies the bitwise operators componentwise.

- If one operand is a scalar and the other is a vector,

- The scalar converts to the element type used by the vector operand.

- The scalar type then widens to a vector that has the same number of components as the vector operand.

- Metal performs the bitwise operation componentwise resulting in a same-size vector.

8. The logical operators [and (&&), or (||)] act upon two operands that are Boolean expressions. The result is a scalar or vector Boolean.

9. The logical unary operator not (!) acts upon one operand that is a Boolean expression. The result is a scalar or vector Boolean.

10. The ternary selection operator (?:) acts upon three operands that are expressions (exp1?exp2:exp3). This operator evaluates the first expression exp1, which must result in a scalar Boolean. If the result is true, the second expression is evaluated; if false, the third expression is evaluated. Metal evaluates only one of the second and third expressions. The second and third expressions can be of any type if:

- The types of the second and third expressions match.

- There is a type conversion for one of the expressions that can make their types match (for more about type conversions, see section 2.12).

- One expression is a vector and the other is a scalar, and the scalar can be widened to the same type as the vector type. The resulting matching type is the type of the entire expression.

11. The ones' complement operator (~) acts upon one operand that needs to be of a scalar or vector integer type. The result is the ones' complement of its operand.

The right-shift (>>) and left-shift (<<) operators act upon all scalar and vector integer type operands. For built-in vector types, Metal applies the operators componentwise. For the right-shift (>>) and left-shift (<<) operators, if the first operand is a scalar, the

rightmost operand needs to be a scalar. If the first operand is a vector, the rightmost operand can be a vector or scalar.

The result of E1 << E2 is E1 left-shifted by the $ \log2(N) $ least significant bits in E2 viewed as an unsigned integer value:

- If E1 is a scalar, N is the number of bits used to represent the data type of E1.

- Or if E1 is a vector, N is the number of bits used to represent the type of E1 elements.

For the left-shift operator, the vacated bits are filled with zeros.

The result of E1 >> E2 is E1 right-shifted by the $ \log2(N) $ least significant bits in E2 viewed as an unsigned integer value:

- If E1 is a scalar, N is the number of bits used to represent the data type of E1.

- Or if E1 is a vector, N is the number of bits used to represent the data type of E1 elements.

For the right-shift operator, if E1 has an unsigned type or if E1 has a signed type and a nonnegative value, the vacated bits are filled with zeros. If E1 has a signed type and a negative value, the vacated bits are filled with ones.

12. The assignment operator behaves as described by the C++17 Specification. For the lvalue = expression assignment operation, if expression is a scalar type and lvalue is a vector type, the scalar converts to the element type used by the vector operand. The scalar type then widens to a vector that has the same number of components as the vector operand. Metal performs the operation componentwise, which results in a same size vector.

Other C++17 operators that are not detailed above such as sizeof(T), unary (&) operator, and comma (,) operator behave as described in the C++17 Specification.

Unsigned integers shall obey the laws of arithmetic modulo 2n, where n is the number of bits in the value representation of that particular size of integer. The result of signed integer overflow is undefined.

For integral operands the divide (/) operator yields the algebraic quotient with any fractional part discarded. (This is often called truncation towards zero.) If the quotient a/b is representable in the type of the result, $ ( a / b ) * b + a \% b $ is equal to a.

## 3.2 Matrix Operators

The arithmetic operators add (+), subtract (-) operate on matrices. Both matrices must have the same numbers of rows and columns. Metal applies the operation componentwise resulting in the same size matrix. The arithmetic operator multiply (*) acts upon:

- a scalar and a matrix

- a matrix and a scalar

- a vector and a matrix

- a matrix and a vector

- a matrix and a matrix

If one operand is a scalar, the scalar value is multiplied to each component of the matrix resulting in the same-size matrix. A right vector operand is treated as a column vector and a left vector operand as a row vector. For vector-to-matrix, matrix-to-vector, and matrix-to-matrix multiplication, the number of columns of the left operand needs to be equal to the number of rows of the right operand. The multiply operation does a linear algebraic multiply, yielding a vector or a matrix that has the same number of rows as the left operand and the same number of columns as the right operand.

The following examples presume these vector, matrix, and scalar variables are initialized. The order of partial sums for the vector-to-matrix, matrix-to-vector, and matrix-to-matrix multiplication operations described below is undefined.

```c

float3 v;

float3x3 m, n;

float a = 3.0f;

```

The matrix-to-scalar multiplication:

```c

float3x3 m1 = m * a;

```

is equivalent to:

```c

m1[0][0] = m[0][0] * a;

m1[0][1] = m[0][1] * a;

m1[0][2] = m[0][2] * a;

m1[1][0] = m[1][0] * a;

m1[1][1] = m[1][1] * a;

m1[1][2] = m[1][2] * a;

m1[2][0] = m[2][0] * a;

m1[2][1] = m[2][1] * a;

m1[2][2] = m[2][2] * a;

```

The vector-to-matrix multiplication:

```c

float3 u = v * m;

```

is equivalent to:

```c

u.x = dot(v, m[0]);

u.y = dot(v, m[1]);

u.z = dot(v, m[2]);

```

The matrix-to-vector multiplication:

```c

float3 u = m * v;

```

is equivalent to:

```c
u.x = m[0].x * v.x + m[1].x * v.y + m[2].x * v.z;
u.y = m[0].y * v.x + m[1].y * v.y + m[2].y * v.z;
u.z = m[0].z * v.x + m[1].z * v.y + m[2].z * v.z;
```

The matrix-to-matrix multiplication:

```c
float3x3 r = m * n; // m, n are float3x3
```

is equivalent to:

```c
r[0].x = m[0].x * n[0].x + m[1].x * n[0].y + m[2].x * n[0].z;
r[0].y = m[0].y * n[0].x + m[1].y * n[0].y + m[2].y * n[0].z;
r[0].z = m[0].z * n[0].x + m[1].z * n[0].y + m[2].z * n[0].z;

r[1].x = m[0].x * n[1].x + m[1].x * n[1].y + m[2].x * n[1].z;
r[1].y = m[0].y * n[1].x + m[1].y * n[1].y + m[2].y * n[1].z;
r[1].z = m[0].z * n[1].x + m[1].z * n[1].y + m[2].z * n[1].z;

r[2].x = m[0].x * n[2].x + m[1].x * n[2].y + m[2].x * n[2].z;
r[2].y = m[0].y * n[2].x + m[1].y * n[2].y + m[2].y * n[2].z;
r[2].z = m[0].z * n[2].x + m[1].z * n[2].y + m[2].z * n[2].z;
```
