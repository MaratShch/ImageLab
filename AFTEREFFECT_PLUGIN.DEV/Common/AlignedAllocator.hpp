#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <memory>
#include <type_traits>
#include <stdexcept>

#if defined(_MSC_VER)
  #include <malloc.h> // _aligned_malloc/_aligned_free
#endif

namespace util
{    

template <typename T, std::size_t Alignment>
class aligned_allocator
{
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of two");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = T const*;
    using reference = T&;
    using const_reference = T const&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U> struct rebind { using other = aligned_allocator<U, Alignment>; };

    aligned_allocator() noexcept = default;
    template <class U> aligned_allocator(aligned_allocator<U, Alignment> const&) noexcept {}

    pointer allocate(size_type n)
    {
        if (n == 0) return nullptr;
        // check multiply overflow
        if (n > static_cast<size_type>(-1) / sizeof(T))
            throw std::bad_array_new_length();

        void* ptr = nullptr;
    #if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
    #else
        // posix_memalign requires alignment be multiple of sizeof(void*)
        int rc = posix_memalign(&ptr, Alignment, n * sizeof(T));
        if (rc != 0) ptr = nullptr;
        if (!ptr) throw std::bad_alloc();
    #endif
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type /*n*/) noexcept
    {
        if (!p) return;
    #if defined(_MSC_VER)
        _aligned_free(p);
    #else
        free(p);
    #endif
    }

    // C++14 allocator must provide construct/destroy (optional in C++17+)
    template <class U, class... Args>
    void construct(U* p, Args&&... args)
    {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) noexcept
    {
        p->~U();
    }

    // equality
    bool operator==(aligned_allocator const&) const noexcept { return true; }
    bool operator!=(aligned_allocator const& other) const noexcept { return !(*this == other); }
};

// Helper: verify pointer alignment
inline bool is_aligned(void const* p, std::size_t alignment) noexcept
{
    return (reinterpret_cast<std::uintptr_t>(p) & (alignment - 1)) == 0;
}

} // namespace util
