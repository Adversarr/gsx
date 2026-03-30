## 4 Address Spaces

The Metal memory model describes the behavior and structure of memory objects in MSL programs. An address space attribute specifies the region of memory from where buffer memory objects are allocated. These attributes describe disjoint address spaces that can also specify access restrictions:

- device (see section 4.1)

- constant (see section 4.2)

- thread (see section 4.3)

- threadgroup (see section 4.4)

- threadgroup_imageblock (see section 4.5)

- ray_data (see section 4.6)

- object_data (see section 4.7)

All OS: Metal 1 and later support the device, threadgroup, constant, and thread attributes. Metal 2.3 and later support ray_data attributes. Metal 3 and later support object_data attributes.

iOS: Metal 2 and later support the threadgroup_imageblock attribute.

macOS: Metal 2.3 and later support the threadgroup_imageblock attribute.

All arguments to a graphics or kernel function that are a pointer or reference to a type need to be declared with an address space attribute. For graphics functions, an argument that is a pointer or reference to a type needs to be declared in the device or constant address space. For kernel functions, an argument that is a pointer or reference to a type needs to be declared in the device, threadgroup, threadgroup_imageblock, or constant address space. The following example introduces the use of several address space attributes. (The threadgroup attribute is supported here for the pointer l_data only if foo is called by a kernel function, as detailed in section 4.4.)

```cpp
void foo(device int *g_data,
    threadgroup int *l_data,
    constant float *c_data)
```

{...}

The address space for a variable at program scope needs to be constant.

Any variable that is a pointer or reference needs to be declared with one of the address space attributes discussed in this section. If an address space attribute is missing on a pointer or reference type declaration, a compilation error occurs.

## 4.1 Device Address Space

The device address space name refers to buffer memory objects allocated from the device memory pool that are both readable and writeable.

A buffer memory object can be declared as a pointer or reference to a scalar, vector or user-defined structure. In an app, Metal API calls allocate the memory for the buffer object, which determines the actual size of the buffer memory.

Some examples are:

```cpp
// An array of a float vector with four components.
device float4 *color;

struct Foo {
    float a[3];
    int b[2];
};

// An array of Foo elements.
device Foo *my_info;
```

Because you always allocate texture objects from the device address space, you don't need the device address attribute for texture types. You cannot directly access the elements of a texture object, so use the built-in functions to read from and write to a texture object (see section 6.12).

## 4.2 Constant Address Space

The constant address space name refers to buffer memory objects allocated from the device memory pool that are read-only. You must declare variables in program scope in the constant address space and initialize them during the declaration statement. The initializer(s) expression must be a core constant expression. (Refer to section 5.20 of the C++17 specification.) The compiler may evaluate a core constant expression at compile time. Variables in program scope have the same lifetime as the program, and their values persist between calls to any of the compute or graphics functions in the program.

constant float samples[] = { 1.0f, 2.0f, 3.0f, 4.0f };

Pointers or references to the constant address space are allowed as arguments to functions.

Writing to variables declared in the constant address space is a compile-time error. Declaring such a variable without initialization is also a compile-time error.

Buffers in the constant address space passed to kernel, vertex, and fragment functions have minimum alignment requirements based on the GPU. See "Minimum constant buffer offset alignment" in the Metal Feature Set Tables for more information.

## 4.3 Thread Address Space

The thread address space refers to the per-thread memory address space. Variables allocated in this address space are not visible to other threads. Variables declared inside a graphics or kernel function are allocated in the thread address space.

```cpp
[[kernel]] void my_kernel(...)
{
    // A float allocated in the per-thread address space
    float x;

    // A pointer to variable x in per-thread address space
    thread float *p = &x;

    ...
}
```

## 4.4 Threadgroup Address Space

A GPU compute unit can execute multiple threads concurrently in a threadgroup, and a GPU can execute a separate threadgroup for each of its compute units.

Threads in a threadgroup can work together by sharing data in threadgroup memory, which is faster on most devices than sharing data in device memory. Use the threadgroup address space to:

- Allocate a threadgroup variable in a kernel, mesh, or object function.

- Define a kernel, fragment, or object function parameter that's a pointer to a type in the threadgroup address space.

See the Metal Feature Set Tables to learn which GPUs support threadgroup space arguments for fragment shaders.

Threadgroup variables in a kernel, mesh, or object function only exist for the lifetime of the threadgroup that executes the kernel. Threadgroup variables in a mid-render kernel function are persistent across mid-render and fragment kernel functions over a tile.

This example kernel demonstrates how to declare both variables and arguments in the threadgroup address space. (The [[threadgroup]] attribute in the code below is explained in section 5.2.1.)

```cpp
kernel void my_kernel(threadgroup float *sharedParameter [[threadgroup(0)]],
    ...)
{
    // Allocate a float in the threadgroup address space.
    threadgroup float sharedFloat;

    // Allocate an array of 10 floats in the threadgroup address space.
    threadgroup float sharedFloatArray[10];

    ...
}
```

For more information about the [[threadgroup(0)]] attribute, see section 5.2.1.

## 4.4.1 SIMD-Groups and Quad-Groups

macOS: Metal 2 and later support SIMD-group functions. Metal 2.1 and later support quad-group functions.

iOS: Metal 2.2 and later support some SIMD-group functions. Metal 2 and later support quad-group functions.

Within a threadgroup, you can divide threads into SIMD-groups, which are collections of threads that execute concurrently. The mapping to SIMD-groups is invariant for the duration of a kernel's execution, across dispatches of a given kernel with the same launch parameters, and from one threadgroup to another within the dispatch (excluding the trailing edge threadgroups in the presence of nonuniform threadgroup sizes). In addition, all SIMD-groups within a threadgroup need to be the same size, apart from the SIMD-group with the maximum index, which may be smaller, if the size of the threadgroup is not evenly divisible by the size of the SIMD-groups.

A quad-group is a SIMD-group with the thread execution width of 4.

For more about kernel function attributes for SIMD-groups and quad-groups, see section 5.2.3.6. For more about threads and thread synchronization, see section 6.9 and its subsections:

- For more about thread synchronization functions, including a SIMD-group barrier, see section 6.9.1.

- For more about SIMD-group functions, see section 6.9.2.

- For more about quad-group functions, see section 6.9.3.

## 4.5 Threadgroup Imageblock Address Space

The threadgroup_imageblock address space refers to objects allocated in threadgroup memory that are only accessible using an imageblock<T, L> object (see section 2.11). A pointer to a user-defined type allocated in the threadgroup_imageblock address space can be an argument to a tile shading function (see section 5.1.9). There is exactly one threadgroup per tile, and each threadgroup can access the threadgroup memory and the imageblock associated with its tile.

- Variables allocated in the threadgroup_imageblock address space in a kernel function are allocated for each threadgroup executing the kernel, are shared by all threads in a threadgroup, and exist only for the lifetime of the threadgroup that executes the kernel. Each thread in the threadgroup uses explicit 2D coordinates to access imageblocks. Do not assume any spatial relationship between the threads and the imageblock. The threadgroup dimensions may be smaller than the tile size.

## 4.6 Ray Data Address Space

All OS: Metal 2.3 and later support ray_data address space.

The ray_data address space refers to objects allocated in memory that is only accessible in an intersection function (see section 5.1.6) for ray tracing. Intersection functions can read and write to a custom payload using the [[payload]] attribute (see Table 5.10) in the ray_data address space. When a shader calls intersect() (see section 6.18.2) with a payload, the system copies the payload to the ray_data address space, calls the intersection function, and when the intersection function returns, it copies the payload back out.

## 4.7 Object Data Address Space

All OS: Metal 3 and later support object_data address space.

Object functions use the object_data address space to pass a payload to a mesh function (see section 5.2.3.9). The object_data address space behaves like the threadgroup address space in that the programming model is explicitly cooperative within the threadgroup. Use the threads in the threadgroup to efficiently compute the payload and varyings. mesh_grid_properties::set_threadgroups_per_grid The payload in the object_data address space is not explicitly bound or initialized, and the implementation manages its lifetime.

## 4.8 Memory Coherency

All OS: Metal 3.2 and later support coherent(device) qualifier and memory_coherence on textures for Apple silicon.

Memory operations in Metal have a concept of a scope of coherency. For a store, the scope of coherence describes the set of threads that may observe the result of the store if you properly synchronize them, and for a load, it describes the set of threads whose stores the load may observe if you properly synchronize them. Metal has the following scopes of coherence:

- Thread coherence — memory writes are only visible to the thread.

- Threadgroup coherence — memory writes are only visible to threads within their threadgroup.

- Device coherence — memory writes are visible to all threads on the device, that is, threads across threadgroups.

Memory in the thread address space has thread coherence, and memory in the threadgroup address space has threadgroup coherence. By default, memory in the device address space has threadgroup coherence.

Metal 3.2 and later support the coherent(device) qualifiers for buffers and memory_coherence_device for textures to indicate that the object has device coherence, that is, memory operations are visible across threads on the device if you properly synchronize them.

```cpp
[[kernel]] void example(
    coherent device float *dptr1,
    coherent(device) device float4 *dptr2,
    texture2d<float, access::read, memory_coherence_device> tex,
    texture2d<float, access::read,
        memory_coherence::memory_coherence_device> tex2)
{...}
```
