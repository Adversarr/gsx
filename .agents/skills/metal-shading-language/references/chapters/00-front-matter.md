<div align="center">

# Metal Shading Language Specification Version 4

</div>

## Contents

**1 Introduction ... 11**

1.1 Purpose of This Document...11

1.2 Organization of This Specification...11

1.3 New in Metal 4...12

1.4 References...12

1.5 Metal and C++17...12

1.5.1 Overloading...12

1.5.2 Templates...13

1.5.3 Preprocessing Directives...13

1.5.4 Restrictions...13

1.6 Compiler and Preprocessor...13

1.6.1 Preprocessor Compiler Options...14

1.6.2 Preprocessor Definitions...14

1.6.3 Math Intrinsics Compiler Options...15

1.6.4 Invariance Compiler Options...17

1.6.5 Optimization Compiler Options...17

1.6.6 Maximum Total Threadgroup Size Option...17

1.6.7 Texture Write Rounding Mode...18

1.6.8 Compiler Options to Enable Modules...18

1.6.9 Compiler Options to Enable Logging...19

1.6.10 Compiler Options Controlling the Language Version...19

1.6.11 Compiler Options to Request or Suppress Warnings...21

1.6.12 Target Conditionals...21

1.6.13 Dynamic Library Linker Options...21

1.6.14 Options for Compiling to GPU Binaries...21

1.6.15 Options for Generating Metal Library Symbol Files...22

1.7 Metal Coordinate Systems...22

**2 Data Types ... 25**

2.1 Scalar Data Types...25

2.2 Vector Data Types...27

2.2.1 Accessing Vector Components...29

2.2.2 Vector Constructors...32

2.2.3 Packed Vector Types...33

2.3 Matrix Data Types...35

2.3.1 Accessing Matrix Components...37

2.3.2 Matrix Constructors...37

2.4 SIMD-group Matrix Data Types...38

2.5 Alignment of Data Types...39

2.6 Atomic Data Types...39

2.7 Pixel Data Types ... 39

2.8 Buffers ... 41

2.9 Textures ... 42

2.9.1 Texture Buffers ... 44

2.10 Samplers ... 45

2.11 Imageblocks ... 48

2.12 Aggregate Types ... 50

2.12.1 Arrays of Textures, Texture Buffers, and Samplers ... 50

2.12.1.1 Array Element Access with [] Operator ... 51

2.12.1.2 Array Capacity ... 51

2.12.1.3 Constructors for Templated Arrays ... 52

2.12.2 Structures of Buffers, Textures, and Samplers ... 53

2.13 Argument Buffers ... 54

2.13.1 Tier 2 Hardware Support for Argument Buffers ... 55

2.14 Uniform Type ... 57

2.14.1 The Need for a Uniform Type ... 57

2.14.2 Behavior of the Uniform Type ... 58

2.14.3 Uniform Control Flow ... 59

2.15 Visible Function Table ... 59

2.16 Function Groups Attribute ... 60

2.17 Ray-Tracing Types ... 61

2.17.1 Ray-Tracing Intersection Tags ... 61

2.17.2 Ray Type ... 65

2.17.3 Intersection Function Table ... 65

2.17.4 Intersection Result Type ... 67

2.17.5 Intersection Result Reference Type ... 68

2.17.6 Intersector Type ... 69

2.17.7 Acceleration Structure Type ... 69

2.17.8 Intersection Query Type ... 71

2.18 Interpolant Type ... 72

2.19 Per-Vertex Values ... 73

2.20 Mesh Shader Types ... 74

2.20.1 Mesh Grid Property Type ... 74

2.20.2 Mesh Type ... 74

2.21 Tensor Types ... 79

2.21.1 Extents Type ... 79

2.21.2 Tensor Type ... 80

2.21.2.1 Host-bound Tensors ... 85

2.21.2.2 Origin-shifted Tensors ... 85

2.21.2.3 Shader-Allocated Tensors ... 86

2.21.3 Cooperative Tensor Type ... 86

2.21.3.1 Layout ... 87

2.21.3.2 Cooperative Tensor ... 89

2.22 Type Conversions and Reinterpreting Data ... 92

2.23 Implicit Type Conversions ... 93

**3 Operators ... 94**

3.1 Scalar and Vector Operators ... 94

3.2 Matrix Operators ... 97

**4 Address Spaces ... 100**

4.1 Device Address Space ... 100

4.2 Constant Address Space ... 101

4.3 Thread Address Space ... 102

4.4 Threadgroup Address Space ... 102

4.4.1 SIMD-Groups and Quad-Groups ... 103

4.5 Threadgroup Imageblock Address Space ... 103

4.6 Ray Data Address Space ... 104

4.7 Object Data Address Space ... 104

4.8 Memory Coherency ... 104

**5 Function and Variable Declarations ... 106**

5.1 Functions ... 106

5.1.1 Vertex Functions ... 107

5.1.1.1 Post-Tessellation Vertex Functions ... 107

5.1.1.2 Patch Type and Number of Control Points Per-Patch ... 107

5.1.2 Fragment Functions ... 108

5.1.3 Compute Functions (Kernels) ... 109

5.1.4 Visible Functions ... 110

5.1.5 Stitchable Functions ... 110

5.1.6 Intersection Functions ... 110

5.1.7 Object Functions ... 112

5.1.8 Mesh Functions ... 112

5.1.9 Tile Functions ... 113

5.1.10 Host Name Attribute ... 114

5.1.11 Templated Qualified Functions ... 114

5.1.12 User Annotation Attribute ... 115

5.2 Function Arguments and Variables ... 115

5.2.1 Locating Buffer, Texture, and Sampler Arguments ... 116

5.2.1.1 Vertex Function Example with Resources and Outputs to Device Memory ... 118

5.2.1.2 Raster Order Groups ... 119

5.2.2 Attributes to Locate Per-Vertex Inputs ... 120

5.2.3 Attributes for Built-in Variables ... 122

5.2.3.1 Vertex Function Input Attributes ... 122

5.2.3.2 Post-Tessellation Vertex Function Input Attributes ... 124

5.2.3.3 Vertex Function Output Attributes ... 125

5.2.3.4 Fragment Function Input Attributes ... 127

5.2.3.5 Fragment Function Output Attributes ... 133

5.2.3.6 Kernel Function Input Attributes ... 135

5.2.3.7 Intersection Function Input Attributes ... 140

5.2.3.8 Intersection Function Output Attributes ... 145

5.2.3.9 Object Function Input Attributes ... 146

5.2.3.10 Mesh Function Input Attributes ... 149

5.2.4 Input Assembly Attribute ... 152

5.2.4.1 Vertex Function Output Example ... 153

5.2.4.2 Fragment Function Input Example ... 154

5.2.4.3 Kernel Function Per-Thread Input Example ... 155

5.3 Storage Class Specifiers ...156

5.4 Sampling and Interpolation Attributes ...156

5.5 Per-Fragment Function Versus Per-Sample Function ...158

5.6 Imageblock Attributes ...158

5.6.1 Matching Data Members of Master and View Imageblocks ...159

5.6.2 Imageblocks and Raster Order Groups ...162

5.6.3 Imageblock Layouts for Fragment Functions ...163

5.6.3.1 Implicit Imageblock Layout for Fragment Functions ...164

5.6.3.2 Explicit Imageblock Layout for Fragment Functions ...164

5.6.4 Imageblock Layouts in Kernel Functions ...165

5.6.5 Aliasing Explicit and Implicit Imageblocks ...166

5.6.6 Imageblocks and Function Constants ...167

5.7 Graphics Function — Signature Matching ...167

5.7.1 Vertex — Fragment Signature Matching ...167

5.7.2 Mesh – Fragment Signature Matching ...171

5.8 Program Scope Function Constants ...172

5.8.1 Specifying Program Scope Function Constants ...172

5.8.1.1 Function Constants to Control Code Paths to Compile ...173

5.8.1.2 Function Constants when Declaring the Arguments of Functions ...174

5.8.1.3 Function Constants for Elements of an Input Assembly Structure ...176

5.8.1.4 Function Constants for Resource Bindings ...177

5.8.1.5 Function Constants for Color Attachments and Raster Order Groups ...178

5.8.1.6 Function Constants with Elements of a Structure ...178

5.9 Program Scope Global Built-ins and Bindings ... 178

5.10 Per-Primitive Viewport and Scissor Rectangle Index Selection ... 180

5.11 Additional Restrictions ... 180

**6 Metal Standard Library ... 181**

6.1 Namespace and Header Files ...181

6.2 Common Functions ...181

6.3 Integer Functions ...182

6.4 Relational Functions ...184

6.5 Math Functions ...185

6.6 Matrix Functions ...191

6.7 SIMD-Group Matrix Functions ...192

6.7.1 Creating, Loading, and Storing Matrix Elements ...192

6.7.2 Matrix Operations ...193

6.8 Geometric Functions ...194

6.9 Synchronization and SIMD-Group Functions ...195

6.9.1 Threadgroup and SIMD-Group Synchronization Functions ...195

6.9.2 SIMD-Group Functions ...196

6.9.2.1 Examples ...202

6.9.3 Quad-Group Functions ...205

6.10 Graphics Functions ...213

6.10.1 Fragment Functions ...213

6.10.1.1 Fragment Functions – Derivatives ...213

6.10.1.2 Fragment Functions — Samples ...214

6.10.1.3 Fragment Functions — Flow Control ...214

6.11 Pull-Model Interpolation ...215

6.12 Texture Functions ...216

6.12.1 1D Texture ...220

6.12.2 1D Texture Array ...222

6.12.3 2D Texture ...224

6.12.3.1 2D Texture Sampling Example ...228

6.12.4 2D Texture Array ...228

6.12.5 3D Texture ...231

6.12.6 Cube Texture ...234

6.12.7 Cube Texture Array ...238

6.12.8 2D Multisampled Texture ...241

6.12.9 2D Multisampled Texture Array ...242

6.12.10 2D Depth Texture ...242

6.12.11 2D Depth Texture Array ...246

6.12.12 2D Multisampled Depth Texture ...249

6.12.13 2D Multisampled Depth Texture Array ...250

6.12.14 Cube Depth Texture ...250

6.12.15 Cube Depth Texture Array ...253

6.12.16 Texture Buffer Functions ...256

6.12.17 Texture Synchronization Functions ...257

6.12.18 Null Texture Functions ...258

6.13 Imageblock Functions ...259

6.13.1 Functions for Imageblocks with Implicit Layout ...259

6.13.2 Functions for Imageblocks with Explicit Layout ...261

6.13.3 Writing an Imageblock Slice to a Region in a Texture ...262

6.14 Pack and Unpack Functions ... 265

6.14.1 Unpack and Convert Integers to a Floating-Point Vector ... 265

6.14.2 Convert Floating-Point Vector to Integers, then Pack the Integers ... 267

6.15 Atomic Functions ... 268

6.15.1 Memory Order ... 268

6.15.2 Thread Scope ... 268

6.15.3 Fence Functions ... 269

6.15.4 Atomic Functions ... 269

6.15.4.1 Atomic Store Functions ... 270

6.15.4.2 Atomic Load Functions ... 270

6.15.4.3 Atomic Exchange Functions ... 271

6.15.4.4 Atomic Compare and Exchange Functions ... 271

6.15.4.5 Atomic Fetch and Modify Functions ... 272

6.15.4.6 Atomic Modify Functions (64 Bits) ... 273

6.16 Encoding Commands for Indirect Command Buffers ... 274

6.16.1 Encoding Render Commands in Indirect Command Buffers ... 274

6.16.2 Encoding Compute Commands in Indirect Command Buffers ... 281

6.16.3 Copying Commands of an Indirect Command Buffer ... 283

6.17 Variable Rasterization Rate ... 284

6.18 Ray-Tracing Functions ... 285

6.18.1 Acceleration Structure Functions ... 285

6.18.2 Intersector Intersect Functions ... 286

6.18.3 Intersector Functions to Control Traversal Behavior ... 298

6.18.4 Intersector Functions for Ray Contribution and Geometry Multiplier ... 301

6.18.5 Intersection Query Functions ... 302

6.18.6 Indirect Instance Descriptors ... 310

6.18.7 Curve Utility Functions ... 311

6.18.8 Intersection Function Buffer Descriptors ... 312

6.19 Logging Functions ... 313

**7 Metal Performance Primitives ... 315**

7.1 Execution Scopes ... 315

7.2 Tensor Operations (TensorOps) ... 316

7.2.1 Matrix Multiplication ... 317

7.2.2 Convolution ... 328

**8 Numerical Compliance ... 331**

8.1 INF, NaN, and Denormalized Numbers ... 331

8.2 Rounding Mode ... 331

8.3 Floating-Point Exceptions ... 331

8.4 ULPs and Relative Error ... 331

8.5 Edge Case Behavior in Flush to Zero Mode ... 338

8.6 Conversion Rules for Floating-Point and Integer Types ... 339

8.7 Texture Addressing and Conversion Rules ... 339

8.7.1 Conversion Rules for Normalized Integer Pixel Data Types ... 339

8.7.1.1 Converting Normalized Integer Pixel Data Types to Floating-Point Values ... 339

8.7.1.2 Converting Floating-Point Values to Normalized Integer Pixel Data Types ... 340

8.7.2 Conversion Rules for Half-Precision Floating-Point Pixel Data Type ... 341

8.7.3 Conversion Rules for Single-Precision Floating-Point Pixel Data Type ... 342

8.7.4 Conversion Rules for 10- and 11-bit Floating-Point Pixel Data Type ... 342

8.7.5 Conversion Rules for 9-bit Floating-Point Pixel Data Type with a 5-bit Exponent ... 342

8.7.6 Conversion Rules for Signed and Unsigned Integer Pixel Data Types ... 343

8.7.7 Conversion Rules for sRGBA and sBGRA Textures ... 343

**9 Appendix ... 345**

9.1 New in Metal 3.2 ...345

## Tables and Figures

Table 1.1. Rounding mode...18

Figure 1. Normalized device coordinate system...23

Figure 2. Viewport coordinate system...23

Figure 3. Normalized 2D texture coordinate system...24

Table 2.1. Metal scalar data types...25

Table 2.2. Size and alignment of scalar data types...26

Table 2.3. Size and alignment of vector data types...28

Table 2.4. Size and alignment of packed vector data types...34

Table 2.5. Size and alignment of matrix data types...36

Table 2.6. Metal pixel data types...40

Table 2.7. Sampler state enumeration values...46

Table 2.8. Imageblock slices and compatible target texture formats...49

Table 2.9. Intersection tags...62

Table 2.10. Mesh template parameter...75

Table 2.11. Mesh vertex attributes...75

Table 2.12. Mesh primitive attributes...76

Table 2.13. Mesh static members...77

Table 2.14 Extents template parameters...79

Table 2.15 Extents member types...79

Table 2.16 Tensor template parameters...81

Table 2.17 Tensor member type definition...82

Table 2.18 Cooperative tensor template parameters...87

Table 2.19 Cooperative tensor type definition...89

Table 5.1. Intersection function primitive types...111

Table 5.2. Attributes for vertex function input arguments...123

Table 5.3. Attributes for post-tessellation vertex function input arguments...124

Table 5.4. Attributes for vertex function return type...125

Table 5.5. Attributes for fragment function input arguments...128

Table 5.6. Attributes for fragment function tile input arguments...132

Table 5.7. Attributes for fragment function return types...133

Table 5.8. Attributes for kernel function input arguments...136

Table 5.9. Attributes for kernel function tile input arguments ... 140

Table 5.10. Attributes for intersection function input arguments ... 141

Table 5.11. Attributes for intersection return types ... 145

Table 5.12. Attributes for object function ... 147

Table 5.13. Attributes for mesh function ... 150

Table 6.1. Common functions in the Metal standard library ... 181

Table 6.2. Integer functions in the Metal standard library ... 182

Table 6.3. Relational functions in the Metal standard library ... 185

Table 6.4. Math functions in the Metal standard library ... 185

Table 6.5. Constants for single-precision floating-point math functions ... 189

Table 6.6. Constants for half-precision floating-point math functions ... 190

Table 6.7. Constants for bfloat floating-point math functions ... 190

Table 6.8. Matrix functions in the Metal standard library ... 191

Table 6.9. SIMD-Group matrix load and stores ... 192

Table 6.10. SIMD-Group operations ... 193

Table 6.11. Geometric functions in the Metal standard library ... 194

Table 6.12. Synchronization compute function in the Metal standard library ... 195

Table 6.13. Memory flag enumeration values for barrier functions ... 196

Table 6.14. SIMD-Group permute functions in the Metal standard library ... 197

Table 6.15. SIMD-Group reduction functions in the Metal standard library ... 200

Table 6.16. Quad-group function in the Metal standard library ... 206

Table 6.17. Quad-group permute functions in the Metal standard library ... 206

Table 6.18. Quad-group reduction functions in the Metal standard library ... 209

Table 6.19. Derivatives fragment functions in the Metal standard library ... 214

Table 6.20. Samples fragment functions in the Metal standard library ... 214

Table 6.21. Fragment flow control function in the Metal standard library ... 215

Table 6.22. Pull-Model Interpolant methods ... 215

Table 6.22. Cube face number ... 235

Table 6.23. Unpack functions ... 266

Table 6.24. Pack functions ... 267

Table 6.25. Atomic operations ... 273

Table 6.26. Atomic modify operations ... 273

Table 6.27. Intersect function ... 287

Table 6.28. Intersect functions input parameters ... 287

Table 6.29. Intersect functions to control traversal ... 298

Table 6.30. Intersection query functions ... 303

Table 6.31. Intersection query functions with max_levels<Count> ... 303

Table 6.32. Intersection query ray value functions ... 304

Table 6.33. Intersection query candidate value functions ... 304

Table 6.34. Intersect query committed value functions ... 305

Table 6.35. Curve utility functions ... 311

Table 7.1 Execution scopes ... 315

Table 7.2 TensorOps ... 316

Table 7.3 MatMul2D data type supported ... 317

Table 7.4 Additional MatMul2D data type supported in OS 26.1 and later ... 318

Table 7.5 MatMul2D descriptor parameters ... 318

Table 7.6 MatMul2D member functions ... 320

Table 7.7 Reduction related functions for cooperative tensors ... 323

Table 7.8 Convolution2d descriptor parameters ... 328

Table 7.9 Convolution run parameter ... 329

Table 8.1. Accuracy of single-precision floating-point operations and functions...331

Table 8.2. Accuracy of single-precision operations and functions with fast math enabled...333

Table 8.3. Accuracy of half-precision floating-point operations and functions...336

Table 8.4. Accuracy of bfloat floating-point operations and functions...338

Table 8.5. Accuracy of bfloat floating-point operations and functions with fast math enabled ... 338

Table 8.6. Conversion to a normalized float value...340

Table 8.7. Conversion from floating-point to a normalized integer value...341

Table 8.8. Conversion between integer pixel data types...343
