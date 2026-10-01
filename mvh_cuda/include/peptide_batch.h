#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mvh_cuda {
// Owned, contiguous inputs. Index i is the batch-local peptide ID used by
// candidates; no pointers to Peptide objects cross the compute interface.
class PeptideBatch {
public:
    void clear(std::size_t expected = 0) {
        if (expected > std::size_t(std::numeric_limits<int>::max()))
            throw std::length_error("Peptide batch exceeds CUDA indexing");
        masses_.clear(); offsets_.clear(); texts_.clear();
        masses_.reserve(expected); offsets_.reserve(expected + 1);
        offsets_.push_back(0);
    }
    PeptideBatch() { clear(); }
    void append(double mass, std::string_view sequence) {
        if (size() >= std::size_t(std::numeric_limits<int>::max()))
            throw std::length_error("Peptide batch exceeds CUDA indexing");
        if (sequence.find('\0') != std::string_view::npos)
            throw std::invalid_argument("Embedded NUL in peptide sequence");
        if (!sequence.empty()) texts_.insert(texts_.end(), sequence.begin(), sequence.end());
        texts_.push_back('\0');
        masses_.push_back(mass);
        offsets_.push_back(texts_.size());
    }
    std::size_t size() const { return masses_.size(); }
    bool empty() const { return masses_.empty(); }
    std::string_view sequence(std::size_t i) const {
        if (i >= size()) throw std::out_of_range("Peptide ID outside batch");
        return {texts_.data() + offsets_[i], std::size_t(offsets_[i + 1] - offsets_[i] - 1)};
    }
    const std::vector<double>& masses() const { return masses_; }
    const std::vector<uint64_t>& offsets() const { return offsets_; }
    const std::vector<char>& texts() const { return texts_; }
private:
    std::vector<double> masses_;
    std::vector<uint64_t> offsets_;
    std::vector<char> texts_;
};
}
