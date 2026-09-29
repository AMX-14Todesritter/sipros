#pragma once
#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mvh_cuda {
// IDs follow first appearance. Owned text remains valid even when restoration
// replaces top-candidate strings; offsets survive growth of the text buffer.
class BatchSequenceIds {
public:
    explicit BatchSequenceIds(std::size_t expectedCount) { reset(expectedCount); }

    // Keep capacity, but never carry IDs or keys across batch boundaries.
    void reset(std::size_t expectedCount) {
        if (expectedCount > std::size_t(std::numeric_limits<int>::max()))
            throw std::length_error("Too many peptide sequences in one batch");
        std::size_t capacity = 8;
        while (expectedCount > capacity / 2) {
            if (capacity > slots_.max_size() / 2)
                throw std::length_error("Sequence ID table capacity exceeded");
            capacity *= 2;
        }
        if (slots_.size() < capacity) slots_.resize(capacity);
        std::fill(slots_.begin(), slots_.end(), -1);
        entries_.clear();
        text_.clear();
        entries_.reserve(expectedCount);
    }

    int get(std::string_view sequence) {
        const auto hash = std::hash<std::string_view>{}(sequence);
        auto slot = findSlot(sequence, hash);
        if (slots_[slot] >= 0) return slots_[slot];
        if (entries_.size() >= std::size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Too many distinct peptide sequences in one batch");
        if (entries_.size() == slots_.size() / 2) {
            grow();
            slot = findSlot(sequence, hash);
        }
        const int id = static_cast<int>(entries_.size());
        const auto offset = text_.size();
        entries_.push_back({hash, offset, sequence.size()});
        try {
            // No terminator is needed: comparisons always use the stored length.
            if (!sequence.empty())
                text_.insert(text_.end(), sequence.begin(), sequence.end());
        } catch (...) {
            entries_.pop_back();
            throw;
        }
        slots_[slot] = id;
        return id;
    }

private:
    struct Entry {
        std::size_t hash, offset, length;
    };

    std::size_t findSlot(std::string_view sequence, std::size_t hash) const {
        const auto mask = slots_.size() - 1;
        auto slot = hash & mask;
        while (slots_[slot] >= 0) {
            const auto& entry = entries_[slots_[slot]];
            if (entry.hash == hash && entry.length == sequence.size() &&
                (sequence.empty() ||
                 std::memcmp(text_.data() + entry.offset, sequence.data(), sequence.size()) == 0))
                break;
            slot = (slot + 1) & mask;
        }
        return slot;
    }

    void grow() {
        if (slots_.size() > slots_.max_size() / 2)
            throw std::length_error("Sequence ID table capacity exceeded");
        std::vector<int> slots(slots_.size() * 2, -1);
        const auto mask = slots.size() - 1;
        for (std::size_t id = 0; id < entries_.size(); ++id) {
            auto slot = entries_[id].hash & mask;
            while (slots[slot] >= 0) slot = (slot + 1) & mask;
            slots[slot] = static_cast<int>(id);
        }
        slots_.swap(slots);
    }

    std::vector<int> slots_;
    std::vector<Entry> entries_;
    std::vector<char> text_;
};
}
