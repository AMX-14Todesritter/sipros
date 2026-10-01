#include "mvh_scan_vector.h"
#include "engine.h"
#include "profiling.h"
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

void MvhScanVector::processPeptideArrayMvh(vector<Peptide *> &vpPeptideArray)
{
    MVH_PROFILE_SCOPE("mvh/batch/process");
    const auto& inputs = mvh_cuda::packPeptideBatch(vpPeptideArray);
    mvh_cuda::assignPeptides2Scans(inputs, vAllPrecursorMassChargeMS2ScanPtrTuples, vpAllMS2Scans);
    mvh_cuda::preprocessingMVH(inputs);
    mvh_cuda::scorePeptidesMVH(vpAllMS2Scans, inputs, vpPeptideArray);
    MVH_PROFILE_SCOPE("mvh/batch/delete_peptides");
    for (auto *peptide : vpPeptideArray) delete peptide;
    vpPeptideArray.clear();
}

void MvhScanVector::searchDatabaseMvh()
{
    MVH_PROFILE_SCOPE("mvh/search/database");
    CLOCKSTART;
    mvh_cuda::SearchPeptideGenerator myProteinDatabase(bScreenOutput);
    vector<Peptide *> vpPeptideArray;
    {
        MVH_PROFILE_SCOPE("mvh/search/load_database");
        myProteinDatabase.loadDatabase();
    }
    {
        MVH_PROFILE_SCOPE("mvh/search/prepare");
        this->preMvh();
    }
    mvh_cuda::beginSearchResults(vpAllMS2Scans);
    // Includes digestion, peptide construction and batch assembly, not scoring.
    MVH_PROFILE_BEGIN(generationRange, "mvh/search/generate_peptides");
    if (myProteinDatabase.getFirstProtein()) {
        auto *currentPeptide = new Peptide;
        while (myProteinDatabase.getNextPeptide(currentPeptide)) {
            // Bound the batch by generated peptides. GPU assignment replaces
            // the former per-peptide CPU query; scan-local order stays intact.
            vpPeptideArray.push_back(currentPeptide);
            currentPeptide = new Peptide;
            if (vpPeptideArray.size() >= size_t(mvh_cuda::peptideBatchSize())) {
                MVH_PROFILE_END(generationRange);
                processPeptideArrayMvh(vpPeptideArray);
                MVH_PROFILE_RESUME(generationRange);
            }
        }
        delete currentPeptide;
        MVH_PROFILE_END(generationRange);
        if (!vpPeptideArray.empty()) processPeptideArrayMvh(vpPeptideArray);
    }
    MVH_PROFILE_END(generationRange);
    CLOCKSTOP;
    MVH_PROFILE_SCOPE("mvh/search/finalize");
    mvh_cuda::finishSearchResults(vpAllMS2Scans);
    this->postMvh();
    MVH::destroyLnTable();
    PeptideUnit::iNumScores = 1;
    cout << "MVH search done.\n" << endl;
}
