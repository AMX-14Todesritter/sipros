#pragma once
#include <cstring>
#include <limits>
#include <memory_resource>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace mvh_cuda {
// IDs follow first appearance, just as in the original string-keyed map.
// Both keys and map nodes belong to this batch's arena. In particular, keys
// never reference top-candidate strings that restoreScoringResults may replace.
class BatchSequenceIds {
public:
    explicit BatchSequenceIds(std::size_t expectedCount) : ids_(&storage_) {
        ids_.reserve(expectedCount);
    }
    int get(std::string_view sequence) {
        const auto found = ids_.find(sequence);
        if (found != ids_.end()) return found->second;
        if (ids_.size() >= std::size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Too many distinct peptide sequences in one batch");
        auto* text = static_cast<char*>(storage_.allocate(sequence.size() + 1, alignof(char)));
        if (!sequence.empty()) std::memcpy(text, sequence.data(), sequence.size());
        text[sequence.size()] = '\0';
        const int id = static_cast<int>(ids_.size());
        ids_.emplace(std::string_view(text, sequence.size()), id);
        return id;
    }
private:
    // Reverse member destruction destroys the map before releasing its arena.
    std::pmr::monotonic_buffer_resource storage_;
    std::pmr::unordered_map<std::string_view, int> ids_;
};
}
