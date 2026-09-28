#include "sequence_ids.h"
#include <iostream>
#include <stdexcept>
#include <string>

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
    std::cout << "PASS: owned keys, duplicates, first-seen IDs, rehash and independent batches\n";
}
