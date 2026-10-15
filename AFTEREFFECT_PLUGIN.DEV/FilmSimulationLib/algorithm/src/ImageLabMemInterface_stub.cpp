// ---------------------------------------------------------------------------
//  ImageLabMemInterface_stub.cpp
//
//  TEST DOUBLE for the plugin's memory pool. NOT part of the plugin.
//
//  The engine obtains its arena through GetMemoryBlock / FreeMemoryBlock, which the
//  ImageLab plugin implements over its own pool. This file provides the same two
//  functions over plain aligned allocation, so the engine and every harness can be
//  built and run outside the plugin.
//
//  LINK EXACTLY ONE of these: this file for standalone tests, or the plugin's real
//  implementation inside the plugin. Linking both is a duplicate-symbol error,
//  which is the desired outcome - it is far better than silently getting the wrong
//  allocator.
// ---------------------------------------------------------------------------

#include "ImageLabMemInterface.hpp"

#include <cstdlib>
#include <cstdint>
#include <vector>

// posix_memalign lives here. Used in preference to std::aligned_alloc because that
// one is C++17 and this project is pinned to C++14.
#if !defined(_MSC_VER)
  #include <malloc.h>
#endif


namespace
{
    // ----------------------------------------------------------------------
    //  Live blocks, indexed by the id handed back to the caller.
    //
    //  A vector rather than a map: ids are dense and small, lookup is a subscript,
    //  and a freed slot is left null so a double free is detectable rather than
    //  undefined.
    //
    //  This is mutable global state, which the ENGINE is forbidden to have. This
    //  file is not the engine - it stands in for the host's allocator, and a real
    //  pool keeps state too. It is not thread safe, which is adequate for the
    //  single-threaded harnesses and is stated here so nobody assumes otherwise.
    // ----------------------------------------------------------------------
    std::vector<void*>& blockTable (void) noexcept
    {
        static std::vector<void*> table;
        return table;
    }
}


// ---------------------------------------------------------------------------
//  GetMemoryBlock
//
//  Returns a non-negative id on success, or -1. The block is aligned to alignHint
//  and its contents are ZEROED, which matters: the arena's padding columns are
//  never written by any stage, and a harness that reads them should see zero rather
//  than whatever the allocator last held.
// ---------------------------------------------------------------------------
int32_t GetMemoryBlock (int32_t bytes, int32_t alignHint, void** ptr) noexcept
{
    if (nullptr == ptr)
        return -1;

    *ptr = nullptr;

    if (bytes <= 0)
        return -1;

    // aligned_alloc requires the size to be a multiple of the alignment, and the
    // alignment to be a power of two and at least sizeof(void*).
    std::size_t align = (alignHint > 0) ? static_cast<std::size_t>(alignHint) : 32u;

    if (align < sizeof(void*))
        align = sizeof(void*);

    const std::size_t size =
        ((static_cast<std::size_t>(bytes) + align - 1u) / align) * align;

    void* p = nullptr;

#if defined(_MSC_VER)
    p = _aligned_malloc(size, align);
#else
    // posix_memalign rather than std::aligned_alloc: the latter is C++17 and this
    // project is C++14. It returns an error code rather than setting errno, and
    // leaves the pointer untouched on failure, hence the explicit null on entry.
    if (0 != posix_memalign(&p, align, size))
        p = nullptr;
#endif

    if (nullptr == p)
        return -1;

    // Zero the block. See the note above on why.
    unsigned char* q = static_cast<unsigned char*>(p);

    for (std::size_t i = 0; i < size; i++)
        q[i] = 0u;

    blockTable().push_back(p);

    *ptr = p;

    return static_cast<int32_t>(blockTable().size() - 1u);
}


// ---------------------------------------------------------------------------
//  FreeMemoryBlock
//
//  Tolerates an out-of-range id and a double free, both by doing nothing. A real
//  pool would log; here silence is right, because the harnesses deliberately call
//  free twice to prove it is safe.
// ---------------------------------------------------------------------------
void FreeMemoryBlock (int32_t blockId) noexcept
{
    if (blockId < 0 || static_cast<std::size_t>(blockId) >= blockTable().size())
        return;

    void* p = blockTable()[static_cast<std::size_t>(blockId)];

    if (nullptr == p)
        return;

#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif

    // Null the slot so a second free on the same id is a no-op rather than a
    // double free.
    blockTable()[static_cast<std::size_t>(blockId)] = nullptr;

    return;
}
