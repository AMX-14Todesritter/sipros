#include "mvh_scan_vector.h"

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

bool MvhScanVector::assignPeptides2Scans(Peptide *currentPeptide)
{
	int i, j;
	bool bAssigned = false;
	vector<pair<int, int>> vpPeptideMassRanges;
	pair<int, int> pairMS2Range;

	GetAllRangeFromMass(currentPeptide->getPeptideMass(), vpPeptideMassRanges);

	for (j = 0; j < (int)vpPeptideMassRanges.size(); j++)
	{
		pairMS2Range = vpPeptideMassRanges.at(j);
		if ((pairMS2Range.first > -1) && (pairMS2Range.second > -1))
		{
			for (i = pairMS2Range.first; i <= pairMS2Range.second; i++)
			{ // vpAllMS2ScanPtrs.at(i)->vpPeptides.push_back(currentPeptide);
				// for DIA and DDA with large isolation window
				tuple<double, int, Peptide *> currentMassChargePeptidePtrTuple =
					{
						get<0>(vAllPrecursorMassChargeMS2ScanPtrTuples[i]),
						get<1>(vAllPrecursorMassChargeMS2ScanPtrTuples[i]),
						currentPeptide};
				get<2>(vAllPrecursorMassChargeMS2ScanPtrTuples[i])
					->vMassChargePeptidePtrTuples.push_back(currentMassChargePeptidePtrTuple);
			}
			bAssigned = true;
		}
	}
	return bAssigned;
}

void MvhScanVector::processPeptideArrayMvh(vector<Peptide *> &vpPeptideArray)
{
	int i, iPeptideArraySize, iScanSize;
	iPeptideArraySize = (int)vpPeptideArray.size();

#pragma omp parallel for shared(vpPeptideArray) private(i) \
	schedule(guided)

	for (i = 0; i < iPeptideArraySize; i++)
	{
		vpPeptideArray.at(i)->preprocessingMVH();
	}

	// every MS2 scans scores their matched peptides

	iScanSize = (int)vpAllMS2Scans.size();
#pragma omp parallel for schedule(guided)

	for (i = 0; i < iScanSize; i++)
	{
		int iThreadId = omp_get_thread_num();
		vpAllMS2Scans[i]->scorePeptidesMVH(psequenceIonMasses.at(iThreadId), _ppdAAforward.at(iThreadId),
										   _ppdAAreverse.at(iThreadId), pSeqs.at(iThreadId));
	}

	// free memory of all peptide objects
	for (i = 0; i < (int)vpPeptideArray.size(); i++)
		delete vpPeptideArray.at(i);

	// empty peptide array
	vpPeptideArray.clear();
}

void MvhScanVector::searchDatabaseMvh()
{
	CLOCKSTART;

	ProteinDatabase myProteinDatabase(bScreenOutput);
	vector<Peptide *> vpPeptideArray;
	Peptide *currentPeptide;
	myProteinDatabase.loadDatabase();
	this->preMvh();
	if (myProteinDatabase.getFirstProtein())
	{
		currentPeptide = new Peptide;
		// get one peptide from the database at a time, until there is no more peptide
		while (myProteinDatabase.getNextPeptide(currentPeptide))
		{
			// assign the pointers of peptides to appropriete MS2Scans
			if (assignPeptides2Scans(currentPeptide))
			{
				// save the new peptide to the array
				vpPeptideArray.push_back(currentPeptide);
				if (currentPeptide->getPeptideMass() > ProNovoConfig::dMaxPeptideMass)
				{
					ProNovoConfig::dMaxPeptideMass = currentPeptide->getPeptideMass();
				}
			}
			else
			{
				delete currentPeptide;
			}
			// create a new peptide for the next iteration
			currentPeptide = new Peptide;
			// when the vpPeptideArray is full
			if (vpPeptideArray.size() >= PEPTIDE_ARRAY_SIZE)
				processPeptideArrayMvh(vpPeptideArray);
		}
		// the last peptide object is an empty object and need to be deleted
		delete currentPeptide;
		// there are still unprocessed peptides in the vpPeptideArray
		// need to process them in the same manner
		// the following code is the same as inside if(vpPeptideArray.size() >= PEPTIDE_ARRAY_SIZE )
		//    cout<<vpPeptideArray.size()<<endl;
		if (!vpPeptideArray.empty())
			processPeptideArrayMvh(vpPeptideArray);
	}
	CLOCKSTOP;

	this->postMvh();
	MVH::destroyLnTable();
	PeptideUnit::iNumScores = 1;
	cout << "MVH search done.\n"
		 << endl;
}
