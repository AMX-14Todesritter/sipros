#include "mvh_scan_vector.h"
#include "engine.h"
#include "peptide_generation.h"

void MvhScanVector::GetAllRangeFromMass(double dPeptideMass, vector<std::pair<int, int>> &vpPeptideMassRanges)
// all ranges of MS2 scans are stored in  vpPeptideMassWindows
{
	int i;
	pair<int, int> pairMS2Range;
	pair<int, int> lastPairRange(-100, -100);
	vector<pair<double, double>> vpPeptideMassWindows;
	vpPeptideMassWindows.clear();
	vpPeptideMassRanges.clear();
	ProNovoConfig::getPeptideMassWindows(dPeptideMass, vpPeptideMassWindows);
	for (i = 0; i < (int)vpPeptideMassWindows.size(); i++)
	{
		pairMS2Range = GetRangeFromMass(vpPeptideMassWindows.at(i).first, vpPeptideMassWindows.at(i).second);
		if ((pairMS2Range.first > -1) && (pairMS2Range.second > -1))
		{
			if ((lastPairRange.first < 0) || (lastPairRange.second < 0))
				lastPairRange = pairMS2Range;
			else
			{
				if (lastPairRange.second > pairMS2Range.first)
					lastPairRange.second = pairMS2Range.second;
				else
				{
					vpPeptideMassRanges.push_back(lastPairRange);
					lastPairRange = pairMS2Range;
				}
			}
		}
	}
	if ((lastPairRange.first > -1) && (lastPairRange.second > -1))
		vpPeptideMassRanges.push_back(lastPairRange);
}

pair<int, int> MvhScanVector::GetRangeFromMass(double lb, double ub)
// lb and ub are lower and upper bounds of acceptable parent mass values
{
	pair<int, int> p;
	int low = 0, high = vpPrecursorMasses.size() - 1, mid;
	double target;
	target = (lb + ub) / 2.0;
	// double lb, ub;  // lower and upper bounds on acceptable parent mass values
	// ub = target + error;
	// lb = target - error;
	while ((high - low) > 1)
	{
		mid = (high + low) / 2;
		if (vpPrecursorMasses[mid] > target)
			high = mid;
		else
			low = mid;
	}

	// Iterate till we get to the first element > than the lower bound
	int ndx = low;
	// cout<<scan_mass_list_.size()<<endl;
	if (vpPrecursorMasses[ndx] >= lb)
	{
		while (ndx >= 0 && vpPrecursorMasses[ndx] >= lb)
			ndx--;
		ndx++;
	}
	else
		while (ndx < (int)vpPrecursorMasses.size() && vpPrecursorMasses[ndx] < lb)
			ndx++;
	if (ndx == (int)vpPrecursorMasses.size() || vpPrecursorMasses[ndx] > ub)
		p = make_pair(-1, -1);
	else
	{
		low = ndx;
		while (ndx < (int)vpPrecursorMasses.size() && vpPrecursorMasses[ndx] <= ub)
			ndx++;
		high = ndx - 1;
		p = make_pair(low, high);
	}

	if (p.first == -1)
		return p;
	if (vpPrecursorMasses[p.first] < lb)
		cerr << "ERROR L " << vpPrecursorMasses[p.first] << " " << lb << endl;
	if (vpPrecursorMasses[p.second] > ub)
		cerr << "ERROR U " << vpPrecursorMasses[p.second] << " " << ub << endl;
	return p;
}

void MvhScanVector::assignPeptides2Scans(const vector<Peptide *> &peptides)
{
    mvh_cuda::assignPeptides2Scans(peptides, vAllPrecursorMassChargeMS2ScanPtrTuples,
                                 vpAllMS2Scans);
}

void MvhScanVector::processPeptideArrayMvh(vector<Peptide *> &vpPeptideArray, uint64_t expectedCandidates)
{
    const auto& inputs = mvh_cuda::packPeptideBatch(vpPeptideArray);
    mvh_cuda::assignPeptides2Scans(inputs, vAllPrecursorMassChargeMS2ScanPtrTuples, vpAllMS2Scans);
    if(expectedCandidates!=std::numeric_limits<uint64_t>::max() &&
       mvh_cuda::assignedCandidateCount()!=expectedCandidates)
        throw std::runtime_error("Candidate batch count changed between lookahead and assignment");
    mvh_cuda::preprocessingMVH(inputs);
    mvh_cuda::scorePeptidesMVH(vpAllMS2Scans, inputs, vpPeptideArray);
    for (auto *peptide : vpPeptideArray) delete peptide;
    vpPeptideArray.clear();
}

void MvhScanVector::searchDatabaseMvh()
{
    mvh_cuda::SearchPeptideGenerator myProteinDatabase(bScreenOutput);
    vector<Peptide *> vpPeptideArray;
    myProteinDatabase.loadDatabase();
    this->preMvh();
    mvh_cuda::beginSearchResults(vpAllMS2Scans);
    // Own any queued raw objects on an error as well as on normal submission.
    struct Cleanup {vector<Peptide*>& objects;~Cleanup(){for(auto *p:objects)delete p;}} cleanup{vpPeptideArray};
    if (myProteinDatabase.getFirstProtein()) {
        if(std::string(mvh_cuda::batchSizingName())=="candidates") {
            mvh_cuda::CandidateBatchCounter counter(vAllPrecursorMassChargeMS2ScanPtrTuples,vpAllMS2Scans);
            const uint64_t limit=mvh_cuda::candidateBatchSize();
            uint64_t associations=0,generated=0,unassigned=0,batches=0,totalCandidates=0;
            auto submit=[&]() {
                if(vpPeptideArray.empty())return;
                std::cout<<"[CUDA candidate batch] peptides="<<vpPeptideArray.size()
                         <<" candidates="<<associations<<" limit="<<limit<<'\n';
                processPeptideArrayMvh(vpPeptideArray,associations);
                totalCandidates+=associations;associations=0;++batches;
            };
            bool exhausted=false;
            while(!exhausted) {
                std::vector<std::unique_ptr<Peptide>> page;
                std::vector<double> masses;
                while(page.size()<65536) {
                    auto peptide=std::make_unique<Peptide>();
                    if(!myProteinDatabase.getNextPeptide(peptide.get())){exhausted=true;break;}
                    masses.push_back(peptide->dPeptideMass);page.push_back(std::move(peptide));
                }
                if(page.empty())break;
                const auto counts=counter.count(masses);generated+=page.size();
                for(size_t i=0;i<page.size();++i) {
                    const auto count=counts[i];
                    if(!count){++unassigned;continue;}
                    if(count>limit)throw std::runtime_error("One peptide produces "+std::to_string(count)+
                        " candidates, exceeding --candidate-batch-size; increase the limit (peptides are not split)");
                    if(associations+count>limit)submit();
                    vpPeptideArray.push_back(page[i].release());associations+=count;
                    if(associations==limit)submit();
                }
            }
            submit();
            std::cout<<"[CUDA candidate batching] generated="<<generated<<" unassigned="<<unassigned
                     <<" candidates="<<totalCandidates<<" batches="<<batches<<" limit="<<limit<<'\n';
        } else {
            auto currentPeptide=std::make_unique<Peptide>();
            while(myProteinDatabase.getNextPeptide(currentPeptide.get())) {
                vpPeptideArray.push_back(currentPeptide.release());
                currentPeptide=std::make_unique<Peptide>();
                if(vpPeptideArray.size()>=size_t(mvh_cuda::peptideBatchSize()))processPeptideArrayMvh(vpPeptideArray);
            }
            if(!vpPeptideArray.empty())processPeptideArrayMvh(vpPeptideArray);
        }
    }
    mvh_cuda::finishSearchResults(vpAllMS2Scans);
    this->postMvh();
    MVH::destroyLnTable();
    PeptideUnit::iNumScores = 1;
    cout << "MVH search done.\n" << endl;
}
