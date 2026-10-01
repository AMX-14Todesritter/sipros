#include "peptide_batch.h"
#include <iostream>
#include <string>
#include <stdexcept>

static void require(bool value) {
    if (!value) throw std::runtime_error("Packed peptide ownership/order contract failed");
}
int main() {
    mvh_cuda::PeptideBatch batch;
    std::string text = "[LDNM~ATK]";
    batch.append(123.125, text);
    text.assign(1000, 'X');
    batch.append(456.25, "");
    for (int i=0;i<10000;++i) batch.append(i, "[PEPTIDE"+std::to_string(i)+"]");
    require(batch.sequence(0) == "[LDNM~ATK]" && batch.sequence(1).empty());
    require(batch.masses()[0] == 123.125 && batch.masses()[1] == 456.25);
    for (int i=0;i<10000;++i) {
        require(batch.sequence(i+2) == "[PEPTIDE"+std::to_string(i)+"]");
        require(batch.masses()[i+2] == i);
    }
    require(batch.offsets().size() == batch.size()+1);
    require(batch.offsets().back() == batch.texts().size());
    for (size_t i=0;i<batch.size();++i)
        require(batch.texts()[batch.offsets()[i+1]-1] == '\0');
    const auto masses = batch.masses().capacity();
    const auto offsets = batch.offsets().capacity();
    const auto bytes = batch.texts().capacity();
    batch.clear(2);
    require(batch.empty() && batch.offsets().size()==1 && batch.offsets()[0]==0);
    batch.append(1, "[M~]"); batch.append(2, "[M~]");
    require(batch.size()==2 && batch.sequence(0)==batch.sequence(1));
    require(batch.masses().capacity()==masses && batch.offsets().capacity()==offsets &&
            batch.texts().capacity()==bytes);
    bool rejected=false;
    try { batch.append(3, std::string("A\0B",3)); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected && batch.size()==2);
    std::cout << "PASS: packed peptide ownership, stable indices, empty sequences and capacity reuse\n";
}
