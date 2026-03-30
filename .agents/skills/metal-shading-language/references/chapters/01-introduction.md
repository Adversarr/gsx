## 1 Introduction

## 1.1 Purpose of This Document

Metal enables you to develop apps that take advantage of the graphics and compute processing power of the GPU. This document describes the Metal Shading Language (MSL), which you will use to write a shader program, which is graphics and data-parallel compute code that runs on the GPU. Shader programs run on different programmable units of the GPU. MSL is a single, unified language that allows tighter integration between the graphics and compute programs. Since MSL is C++-based, you will find it familiar and easy to use.

MSL works with the Metal framework, which manages the execution and optionally the compilation of the Metal programs. Metal uses clang and LLVM so you get a compiler that delivers optimized performance on the GPU.

## 1.2 Organization of This Specification

This document is organized into the following chapters:

- This chapter, "Introduction," is an introduction to this document that covers the similarities and differences between Metal and C++17. It also details the options for the Metal compiler, including preprocessor directives, options for math intrinsics, and options for controlling optimization.

- "Data Types" lists the Metal data types, including types that represent vectors, matrices, buffers, textures, and samplers. It also discusses type alignment and type conversion.

- "Operators" lists the Metal operators.

- "Address Spaces" describes disjoint address spaces for allocating memory objects with access restrictions.

- "Function and Variable Declarations" details how to declare functions and variables, with optional attributes that specify restrictions.

- "Metal Standard Library" defines a collection of built-in Metal functions.

- "Numerical Compliance" describes requirements for representing floating-point numbers, including accuracy in mathematical operations.

iOS and macOS support for features (functions, enumerations, types, attributes, or operators) described in this document is available since Metal 1, unless otherwise indicated.

For the rest of this document, the abbreviation X.Y stands for "Metal version X.Y"; for example, 2.1 indicates Metal 2.1. Please note that though a feature is supported in MSL shading language, it may not be supported on all GPUs. Please refer to the Metal Feature Set Tables at developer.apple.com.

## 1.3 New in Metal 4

Metal 4 introduces the following new features:

- C++17 based (section 1.5)

- Sampler LOD bias, minimum and maximum reduction (section 2.10)

- Intersection Function Buffers (section 2.17.1, 5.1.6, 5.2.3.7, 6.18.2, 6.18.4, and 6.18.8)

- Per-Vertex values (section 2.19)

- Tensors (section 2.21)

- User annotations (section 5.1.12)

- Texture atomics for cube and cube array textures (section 6.12.6 and 6.12.7)

- Pack and unpack of snorm10a2 (section 6.14)

- Indirect command buffer support for raster and depth stencil states (section 6.16.1)

- Metal Performance Primitives (section 7)

## 1.4 References

Metal

Here is a link to the Metal documentation on apple.com:

https://developer.apple.com/documentation/metal

## 1.5 Metal and C++17

Starting in Metal 4, the Metal programming language is a C++17-based specification with extensions and restrictions. Refer to the C++17 specification (also known as ISO/IEC 14882:2017) for a detailed description of the language grammar. Prior language versions of Metal are a C++14-based specification with extensions and restrictions.

This section and its subsections describe the modifications and restrictions to the C++17 and C++14 language supported in Metal.

For more about Metal preprocessing directives and compiler options, see section 1.6 of this document.

## 1.5.1 Overloading

Metal supports overloading, as defined by section 13 of the C++17 and C++14 specification. Metal extends the function overloading rules to include the address space attribute of an argument. You cannot overload Metal graphics and kernel functions. (For a definition of graphics and kernel functions, see section 5.1 of this document.)

## 1.5.2 Templates

Metal supports templates, as defined by section 14 of the C++17 and C++14 specification.

## 1.5.3 Preprocessing Directives

Metal supports the preprocessing directives, as defined by section 16 of the C++17 and C++14 Specification.

## 1.5.4 Restrictions

All OS: Metal 3.2 and later support lambda expressions.

The following C++17 features are not available in Metal (section numbers in this list refer to the C++17 Specification):

- lambda expressions (section 5.1.2) prior to Metal 3.2

- dynamic_cast operator (section 5.2.7)

- type identification (section 5.2.8)

- new and delete operators (sections 5.3.4 and 5.3.5)

- noexcept operator (section 5.3.7)

- goto statement (section 6.6)

- register, thread_local storage attributes (section 7.1.1)

- virtual function attribute (section 7.1.2)

- derived classes (section 10, section 11)

- exception handling (section 15)

Do not use the C++ standard library in Metal code. Instead, Metal has its own standard library, as discussed in section 5 of this document.

Metal restricts the use of pointers:

- You must declare arguments to Metal graphics and kernel functions that are pointers with the Metal device, constant, threadgroup, threadgroup_imageblock, object_data, or ray_data address space attribute. (For more about Metal address space attributes, see section 4 of this document.)

- Metal 2.3 and later support function pointers.

Metal supports recursive function calls (C++ section 5.2.2, item 9) in compute (kernel) context starting with Metal 2.4.

You can't call a Metal function main.

## 1.6 Compiler and Preprocessor

You can use the Metal compiler online (with the appropriate APIs to compile Metal sources) or offline. You can load Metal sources that are compiled offline as binaries, using the appropriate Metal APIs.

This section explains the compiler options supported by the Metal compiler and categorizes them as preprocessor options, options for math intrinsics, options that control optimization, miscellaneous compilation options, and linking.

## 1.6.1 Preprocessor Compiler Options

The following options control the Metal preprocessor that runs on each program source before actual compilation:

-D name

Predefine name as a macro, with definition 1.

-D name=definition

Metal tokenizes and processes the contents of definition as if they appear in a #define directive. This option allows you to compile Metal code to enable or disable features. You may use this option multiple times, and the preprocessor processes the definitions in the order in which they appear.

-I dir

Add the directory dir to the search path of directories for header files. This option is only available for the offline compiler.

## 1.6.2 Preprocessor Definitions

The Metal compiler sets a number of preprocessor definitions by default, including:

__METAL_VERSION__ // Set to the Metal language revision

__METAL_MACOS__ // Set if compiled with the macOS Metal language

__METAL_IOS__ // Set if compiled with the iOS Metal language

__METAL__ // Set if compiled with the unified Metal language

// Set with -std=metal3.0 or above

You can use definitions to conditionally apply shading language features that are only available on later language version (see section 1.6.10 Compiler Options Controlling the Language Version).

The version number is MajorMinorPatch. For example, for Metal 1.2, patch 0, __METAL_VERSION__ is 120; for Metal 2.1, patch 1, __METAL_VERSION__ is 211.

To conditionally include code that uses features introduced in Metal 2, you can use the preprocessor definition in code, as follows:

```cpp

#if __METAL_VERSION__ >= 200

// Code that requires features introduced in Metal 2.

#endif

```

## 1.6.3 Math Intrinsics Compiler Options

The following section describes options to control compiler behavior regarding floating-point arithmetic, trading off between speed and correctness.

For more about math functions, see section 6.5. For more about the relative errors of ordinary and fast math functions, see section 8.4.

The options enable or disable the optimizations for floating-point arithmetic that may violate the IEEE 754 standard. They also enable or disable the high precision variant of math functions for single precision floating-point scalar and vector types.

The fast math optimizations for floating-point arithmetic include:

- No NaNs: Allow optimizations to assume the arguments and result are not NaN (not a number).

- No INFs: Allow optimizations to assume the arguments and result are not positive or negative infinity.

- No Signed Zeroes: Allow optimizations to treat the sign of a zero argument or result as insignificant.

- Allow Reciprocal: Allow optimizations to use the reciprocal of an argument rather than perform a division.

- Allow Reassociation: Allow algebraically equivalent transformations, such as reassociating floating-point operations that may dramatically change the floating-point results.

- Allow Contract: Allow floating-point contraction across statements. For example, allow fusing a multiply followed by an addition into a single fused-multiply-add.

Metal supports the following options beginning with Xcode 16 and Metal Developer Tools for Windows 5 (SDK supporting iOS 18 or macOS 15):

-fmetal-math-fp32-functions=<fast|precise>

This option sets the single-precision floating-point math functions described in section 6.5 to call either the fast or precise version. The default is fast. For Apple silicon, starting with Apple GPU Family 4, the math functions honor INF and NaN.

-fmetal-math-mode=<fast, relaxed, safe>

This option sets how aggressive the compiler can be with floating-point optimizations. The default is fast.

If you set the option to fast, it lets the compiler make aggressive, potentially lossy assumptions about floating-point math. These include no NaNs, no INFs, no signed zeros, allow reciprocal, allow reassociation, and FP contract to be fast.

If you set the option to relaxed, it lets the compiler make aggressive, potentially lossy assumptions about floating-point math, but honors INFs and NaNs. These include no signed zeros, allow reciprocal, allow reassociation, and FP contract to be fast. This supports Apple silicon.

If you set the option to safe, it disables unsafe floating-point optimizations by preventing the compiler from making any transformations that might affect the results. This sets the FP contract to on.

Metal supports the following legacy options:

-ffast-math

Equivalent to -fmetal-math-fp32-functions=fast and -fmetal-math-mode=fast.

-fno-fast-math

Equivalent to -fmetal-math-fp32-functions=precise and -fmetal-math-mode=safe.

When utilizing fast math in your program, it is important to understand that the compiler can assume certain properties and make optimizations accordingly. For example, the use of fast math asserts that the shader will never generate INF or NaN. If the program has an expression X/Y, the compiler can assume Y is never zero as this could potentially result in positive/negative infinite or NaN, depending on the value of X. If Y can be zero, you would have an undefined program if compiled with fast math.

The #pragma metal fp pragmas allow you to specify floating-point options for a source code section.

The following pragma has the same semantics to allow you to specify precise floating-point semantics and floating-point exception behavior for a source code section. It can only appear in file or namespace scope, within a language linkage specification, or at the start of a compound statement (excluding comments). When using it within a compound statement, the pragma is active within the scope of the compound statement:

```cpp

#pragma METAL fp math_mode([relaxed | safe | fast])

```

By default, the compiler allows floating-point contractions. For example, a*b+c may be converted to a single fused-multiply-add. These contractions could lead to computation differences if other expressions are not contracted. To disable allowing the compiler to perform contractions, pass the following option:

-ffp-contract=off

The compiler also supports controlling contractions with the following pragma:

#pragma METAL fp contract([off | on | fast])

Using off disables contractions, on allows contractions within a statement, and fast allows contractions across statements. You can also use:

#pragma STDC FP_CONTRACT OFF

## 1.6.4 Invariance Compiler Options

If you are building with an SDK that supports iOS 14 or macOS 11, you need to pass the following option to support vertex invariance:

-fpreserve-invariance

Preserve invariant for computations marked with [[invariant]] in vertex shaders. If not set, [[invariant]] is ignored.

In previous versions of Metal, [[invariant]] was a best-effort analysis to mark which operations need to be invariant and may fail in certain cases. This is replaced with a conservative invariant model where the compiler marks operations that do not go into an invariant calculation. This guarantees anything that is in an invariant calculation remains invariant. This option may reduce performance as it may prevent certain optimizations to preserve invariance.

## 1.6.5 Optimization Compiler Options

These options control the optimization level of the compiler:

-O2

Optimize for performance (default).

-Os

Like -O2 with extra optimizations to reduce code size.

## 1.6.6 Maximum Total Threadgroup Size Option

All OS: Metal 3 and later support maximum total threadgroup size option.

This option specifies the number of threads (value) in a threadgroup for every function in the translation unit:

-fmax-total-threads-per-threadgroup=<value>

The attribute [[max_total_threads_per_threadgroup]] function attribute described in section 5.1.3, section 5.1.7, and section 5.1.8 takes precedence over the compile option. The value must fit within 32 bits.

This option is useful for setting the option to enable functions compiled for a dynamic library to be compatible with a PSO.

## 1.6.7 Texture Write Rounding Mode

Configure the rounding mode for texture writes to floating-point pixel types by setting the -ftexture-write-rounding-mode compiler flag to one of the options in Table 1.1.

<div align="center">

Table 1.1. Rounding mode

</div>

<table border="1"><tr><td>Rounding mode</td><td>Description</td></tr><tr><td>native (default)</td><td>Texture writes use the hardware's native rounding strategy.</td></tr><tr><td>rte<br>All OS: Metal 2.3 and later</td><td>Texture writes round to the nearest even number.</td></tr><tr><td>rtz<br>All OS: Metal 2.3 and later</td><td>Texture writes round toward zero.</td></tr></table>

The -ftexture-write-rounding-mode flag is available for these SDKs:

- macOS 11 and later

- iOS 14 and later

For more information about which GPU families support rounding modes other than native, see the Metal Feature Set Tables.

## 1.6.8 Compiler Options to Enable Modules

The compiler supports multiple options to control the use of modules. These options are only available for the offline compiler:

-fmodules

Enable the modules feature.

-fimplicit-module-maps

Enable the implicit search for module map files named module.modulemap or a similar name. By default, -fmodules enables this option. (The compiler option -fno-implicit-module-maps disables this option.)

-fno-implicit-module-maps

Disable the implicit search for module map files named module.modulemap. Module map files are only loaded if they are explicitly specified with -fmodule-map-file or transitively used by another module map file.

-fmodules-cache-path=<directory>

Specify the path to the modules cache. If not provided, the compiler selects a system-appropriate default.

-fmodule-map-file=<file>

Load the specified module map file, if a header from its directory or one of its subdirectories is loaded.

If you are building with an SDK that supports iOS 16 or macOS 13, -fmodules has the following additional options:

-fmodules=[mode]

Supported values for modes are:

stdlib: Enable the modules feature but restrict the search for module maps to the Metal standard library. Enabled by default with an SDK that supports iOS 16 or macOS 13.

all: Enable the modules feature (equivalent to -fmodules).

none: Disable the modules feature.

## 1.6.9 Compiler Options to Enable Logging

All OS: Metal 3.2 and later support logging for Apple silicon.

You need to provide the following compiler option to enable logging (see section 6.19) during compilation:

-fmetal-enable-logging

## 1.6.10 Compiler Options Controlling the Language Version

The following option controls the version of the unified graphics and computing language accepted by the compiler:

-std=

Determine the language revision to use. A value for this option must be provided, which must be one of:

- ios-metal1.0: Supports the unified graphics and computing language revision 1 programs for iOS 8. [[deprecated]]

- ios-metal1.1: Supports the unified graphics and computing language revision 1.1 programs for iOS 9.

- ios-metal1.2: Supports the unified graphics and computing language revision 1.2 programs for iOS 10.

- ios-metal2.0: Supports the unified graphics and computing language revision 2 programs for iOS 11.

- ios-metal2.1: Supports the unified graphics and computing language revision 2.1 programs for iOS 12.

- ios-metal2.2: Supports the unified graphics and computing language revision 2.2 programs for iOS 13.

- ios-metal2.3: Supports the unified graphics and computing language revision 2.3 programs for iOS 14.

- ios-metal2.4: Supports the unified graphics and computing language revision 2.4 programs for iOS 15.

- macos-metal1.1 or osx-metal1.1: Supports the unified graphics and computing language revision 1.1 programs for macOS 10.11.

- macos-metal1.2 or osx-metal1.2: Supports the unified graphics and computing language revision 1.2 programs for macOS 10.12.

- macos-metal2.0 or osx-metal2.0: Supports the unified graphics and computing language revision 2 programs for macOS 10.13.

- macos-metal2.1: Supports the unified graphics and computing language revision 2.1 programs for macOS 10.14.

- macos-metal2.2: Supports the unified graphics and computing language revision 2.2 programs for macOS 10.15.

- macos-metal2.3: Supports the unified graphics and computing language revision 2.3 programs for macOS 11.

- macos-metal2.4: Supports the unified graphics and computing language revision 2.4 programs for macOS 12.

Note that macos-* is available in macOS 10.13 SDK and later.

As of iOS 16, macOS 13, and tvOS 16, Metal has unified the shading language between the platforms:

- metal3.0: Supports the unified graphics and computing language revision 3 programs for iOS 16, macOS 13, and tvOS 16.

- metal3.1: Supports the unified graphics and computing language revision 3.1 programs for iOS 17, macOS 14, tvOS 17, and visionOS 1.

Only Apple Silicon supports new features in language standard 3.2 and above:

- metal3.2: Supports the unified graphics and computing language revision 3.2 programs for iOS 18, macOS 15, tvOS 18, and visionOS 2.

- metal4.0: Supports the unified graphics and computing language revision 4 programs for iOS 26, macOS 26, tvOS 26, and visionOS 26.

## 1.6.11 Compiler Options to Request or Suppress Warnings

The following options are available:

-Werror

Make all warnings into errors.

-w

Inhibit all warning messages.

## 1.6.12 Target Conditionals

Metal defines several macros which one can use to determine what platform the shader is running on. The following macros are defined in <TargetConditionals.h>:

TARGET_OS_MAC : Generated code runs under Mac OS X variant

TARGET_OS_OSX : Generated code runs under OS X devices

TARGET_OS_IPHONE : Generated code for firmware, devices or simulator

TARGET_OS_IOS : Generated code runs under iOS

TARGET_OS_TV : Generated code runs under tvOS

TARGET_OS_MACCATALYST : Generated code runs under macOS

TARGET_OS_SIMULATOR : Generated code runs under a simulator

TARGET_OS_VISION : Generated code runs under visionOS

(Available in SDKs in late 2023)

Note that this header is not part of <metal_stdlib>.

## 1.6.13 Dynamic Library Linker Options

The Metal compiler driver can pass options to the linker. Here is a brief description of some of these options. See the Metal linker for more information:

-dynamiclib

Specify that the output is a dynamic library.

-install_name

Used with -dynamiclib to specify the location of where the dynamic library is expected be installed and found by the loader. Use with @executable_path and @loader_path.

## 1.6.14 Options for Compiling to GPU Binaries

The following options are available for compiling to a GPU binary if you are building with an SDK that supports iOS 16 or macOS 13:

-arch [architecture]

Specify the architecture to build for.

-gpu-family [gpu family name]

Specify the architectures associated with the MTLGPUFamily to build for. See MTLGPUFamily in Metal API for the list of available families.

-N [descriptor.mtlp-json]

Specify the pipeline descriptors in Metal script format. The descriptor files must end in .mtlp-json.

## 1.6.15 Options for Generating Metal Library Symbol Files

If you are building with an SDK that supports iOS 15 or macOS 12, the following option is available to generate a Metal library symbol file:

-frecord-sources

Enable the compiler to store source information into the AIR or Metal library file (.metallib).

-frecord-sources=flat

Enable the compiler to store source information if generating an AIR file. Enable the compiler to store the source information in a symbol companion file (.metallibsym) if generating a Metal Library file.

See Generating and loading a Metal library symbol file at developer.apple.com for more information.

## 1.7 Metal Coordinate Systems

Metal defines several standard coordinate systems to represent transformed graphics data at different stages along the rendering pipeline.

A four-dimensional homogenous vector (x, y, z, w) specifies a three-dimensional point in clip space coordinates. A vertex shader generates positions in clip-space coordinates. Metal divides the x, y, and z values by w to convert clip-space coordinates into normalized device coordinates.

Normalized device coordinates use a left-handed coordinate system (see Figure 1) and map to positions in the viewport. These coordinates are independent of viewport size. The lower-left corner of the viewport is at an (x, y) coordinate of (-1.0, -1.0) and the upper corner is at (1.0, 1.0). Positive-z values point away from the camera ("into the screen"). The visible portion of the z coordinate is between 0.0 and 1.0. The Metal rendering pipeline clips primitives to this box.

<div align="center">

Figure 1. Normalized device coordinate system

</div>

![Image 12-0](imgs/cropped_page12_idx0.jpg)

The rasterizer stage transforms normalized-device coordinates (NDC) into viewport coordinates (see Figure 2). The (x,y) coordinates in this space are measured in pixels, with the origin in the top-left corner of the viewport and positive values going to the right and down. You specify viewports in this coordinate space, and the Metal maps NDC coordinates to the extents of the viewport.

If you are using variable rasterization rate (see Section 6.15), then the viewport coordinate system is a logical coordinate system independent of the render target's physical layout. A rate map determines the relationship between coordinates in this logical coordinate system (sometimes called screen space) and pixels in the render targets (physical coordinates).

<div align="center">

Figure 2. Viewport coordinate system

</div>

![Image 12-1](imgs/cropped_page12_idx1.jpg)

Texture coordinates use a similar coordinate system to viewport coordinates. Texture coordinates can also be specified using normalized texture coordinates. For 2D textures, normalized texture coordinates are values from 0.0 to 1.0 in both x and y directions, as seen in Figure 3. A value of (0.0,0.0) specifies the pixel at the first byte of the image data (the top-left corner of the image). A value of (1.0,1.0) specifies the pixel at the last byte of the image data (the bottom-right corner of the image).

<div align="center">

Figure 3. Normalized 2D texture coordinate system

</div>

![Image 13-2](imgs/cropped_page13_idx2.jpg)
