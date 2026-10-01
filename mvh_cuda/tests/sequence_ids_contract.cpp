#include "sequence_ids.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <random>
#include <unordered_map>
#include <vector>

static void require(bool ok) {
    if (!ok) throw std::runtime_error("Sequence ID ownership/order contract failed");
}
int main() {
    mvh_cuda::BatchSequenceIds ids(1);
    std::string source(200, 'A');
    require(ids.get(source) == 0);
    source.assign(200, 'B'); // Existing IDs must not borrow the caller's storage.
    require(ids.get(source) == 1);
    require(ids.get(std::string(200, 'A')) == 0);
    require(ids.get("") == 2);
    for (int i = 0; i < 10000; ++i)
        require(ids.get("peptide-" + std::to_string(i)) == i + 3);
    for (int i = 9999; i >= 0; --i)
        require(ids.get("peptide-" + std::to_string(i)) == i + 3);
    mvh_cuda::BatchSequenceIds nextBatch(1);
    require(nextBatch.get(source) == 0);
    // Force one probe chain, including wraparound, without triggering growth.
    mvh_cuda::BatchSequenceIds collisions(0);
    std::vector<std::string> collidingKeys;
    for (int i = 0; collidingKeys.size() < 4; ++i) {
        auto key = "collision-" + std::to_string(i);
        if ((std::hash<std::string_view>{}(key) & 7) == 7)
            collidingKeys.push_back(std::move(key));
    }
    for (int i = 0; i < 4; ++i) require(collisions.get(collidingKeys[i]) == i);
    for (int i = 3; i >= 0; --i) require(collisions.get(collidingKeys[i]) == i);

    // Differential coverage: duplicates, binary keys, common prefixes, and
    // repeated growth with no capacity estimate (including an empty first key).
    mvh_cuda::BatchSequenceIds growing(0);
    std::unordered_map<std::string, int> reference;
    std::vector<std::string> keys = {"", std::string("A\0B", 3), "A", "AB"};
    std::mt19937 random(12345);
    for (int i = 0; i < 50000; ++i) {
        std::string key(30, 'A');
        key += std::to_string(random() % 20000);
        if (i % 7 == 0) key.append(1000, 'Z');
        keys.push_back(std::move(key));
    }
    for (const auto& key : keys) {
        const int expected = static_cast<int>(reference.size());
        const auto inserted = reference.emplace(key, expected);
        require(growing.get(key) == inserted.first->second);
    }
    for (auto it = keys.rbegin(); it != keys.rend(); ++it)
        require(growing.get(*it) == reference.at(*it));
    // Reusing a large workspace for smaller/empty batches must restart IDs and
    // remove all old keys, even when hash slots and byte capacity are retained.
    for (const std::size_t expected : {std::size_t(0), std::size_t(1), std::size_t(60000)}) {
        growing.reset(expected);
        require(growing.get("new-first") == 0);
        require(growing.get(keys.back()) == 1);
        require(growing.get("") == 2);
        require(growing.get("new-first") == 0);
        reference.clear();
        reference.emplace("new-first", 0);
        reference.emplace(keys.back(), 1);
        reference.emplace("", 2);
        for (const auto& key : keys) {
            const auto inserted = reference.emplace(key, static_cast<int>(reference.size()));
            require(growing.get(key) == inserted.first->second);
        }
    }
    // A live sequence keeps its ID across batches, while dead keys/IDs are
    // reclaimed. Include sparse IDs, binary strings, growth and empty Top.
    mvh_cuda::BatchSequenceIds persistent(0);
    std::unordered_map<std::string, int> live;
    for (int batch = 0; batch < 100; ++batch) {
        for (const auto& item : live) require(persistent.get(item.first) == item.second);
        auto current = live;
        for (int i = 0; i < 200; ++i) {
            const auto key = std::string("A\0", 2) + std::to_string(random() % 2000);
            const int id = persistent.get(key);
            auto inserted = current.emplace(key, id);
            require(inserted.first->second == id);
        }
        std::vector<int> keep;
        live.clear();
        std::unordered_map<int, std::string> unique;
        for (const auto& item : current) {
            require(unique.emplace(item.second, item.first).second);
            if (random() % 5 == 0) {
                live.insert(item);
                keep.push_back(item.second);
                keep.push_back(item.second); // Several scans may retain one sequence.
            }
        }
        persistent.retain(keep);
        require(persistent.size() == live.size());
    }
    persistent.retain({});
    require(persistent.size() == 0);
    require(persistent.get("new dataset") == 0);
    std::cout << "PASS: owned keys, duplicates, first-seen IDs, rehash and independent batches\n";
}
