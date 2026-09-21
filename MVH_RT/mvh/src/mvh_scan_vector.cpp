#include "mvh_scan_vector.h"

MvhScanVector::MvhScanVector(const string &sFT2FilenameInput, const string &sOutputDirectory,
							 const string &sConfigFilename, bool bScreenOutput)
{
	unsigned int n;
	vector<string> vsSingleResidueNames = ProNovoConfig::vsSingleResidueNames;
	vector<double> vdSingleResidueMasses = ProNovoConfig::vdSingleResidueMasses;
	sFT2Filename = sFT2FilenameInput;
	sConfigFile = sConfigFilename;
	// mass_w = ProNovoConfig::getParentMassWindows();
	setOutputFile(sFT2FilenameInput, sOutputDirectory);
	this->bScreenOutput = bScreenOutput;
	for (n = 0; n < vsSingleResidueNames.size(); ++n)
		mapResidueMass[vsSingleResidueNames[n][0]] = vdSingleResidueMasses[n];

	iOpenMPTaskNum = 0;
}

MvhScanVector::~MvhScanVector()
{
	// the destructors will free memory from vpAllMS2Scans
	vector<MS2Scan *>::iterator it;
	for (it = vpAllMS2Scans.begin(); it != vpAllMS2Scans.end(); ++it)
	{
		delete (*it);
	}
	vpAllMS2Scans.clear();
	vpPrecursorMasses.clear();
}

void MvhScanVector::preMvh()
{
	num_max_threads = omp_get_max_threads();
	for (int i = 0; i < num_max_threads; ++i)
	{
		_ppdAAforward.push_back(new vector<double>());
		_ppdAAreverse.push_back(new vector<double>());
		psequenceIonMasses.push_back(new vector<double>());
		pSeqs.push_back(new vector<char>());
	}
}

void MvhScanVector::postMvh()
{
	for (int i = 0; i < num_max_threads; ++i)
	{
		delete _ppdAAforward.at(i);
		delete _ppdAAreverse.at(i);
		delete psequenceIonMasses.at(i);
		delete pSeqs.at(i);
	}
	_ppdAAforward.clear();
	_ppdAAreverse.clear();
	psequenceIonMasses.clear();
	pSeqs.clear();
}
