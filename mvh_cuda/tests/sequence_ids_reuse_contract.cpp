#include "sequence_ids.h"
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>

// Count actual heap allocations, including vector backing storage. A warmed
// workspace must service equal/smaller batches without allocating again.
static std::size_t allocations = 0;
void* operator new(std::size_t size) {
    if (void* p = std::malloc(size ? size : 1)) { ++allocations; return p; }
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

int main() {
    std::vector<std::string> keys;
    for (int i = 0; i < 4096; ++i)
        keys.push_back(std::string(100, 'A') + std::to_string(i));
    mvh_cuda::BatchSequenceIds ids(keys.size());
    for (const auto& key : keys) ids.get(key);
    const auto before = allocations;
    for (int batch = 0; batch < 8; ++batch) {
        const auto count = batch % 2 ? keys.size() : keys.size() / 4;
        ids.reset(count);
        for (std::size_t i = 0; i < count; ++i)
            if (ids.get(keys[count - 1 - i]) != static_cast<int>(i)) return 1;
        for (std::size_t i = 0; i < count; ++i)
            if (ids.get(keys[count - 1 - i]) != static_cast<int>(i)) return 2;
    }
    if (allocations != before) return 3;
    std::cout << "PASS: no allocations for warmed equal/smaller batches; IDs restart\n";
}
