<<<<<<< HEAD
#include "profiling.h"
=======
#include <stdexcept>
>>>>>>> 3b5bec1 (candidate stratege update)
#include "mvh_scan_vector.h"
#include "engine.h"
#include "SiprosReader.h"

void MvhScanVector::setOutputFile(const string &sFT2FilenameInput, const string &sOutputDirectory)
{
	std::filesystem::path inPath = sFT2FilenameInput;
	std::filesystem::path outPath = sOutputDirectory;
	// get file name without extension and path
	std::string baseName = inPath.stem().string();
	string searchName;
	if ((ProNovoConfig::getSearchName() == "") || (ProNovoConfig::getSearchName() == "Null") || (ProNovoConfig::getSearchName() == "NULL") || (ProNovoConfig::getSearchName() == "null"))
		searchName = "Null";
	else
	{
		searchName = ProNovoConfig::getSearchName();
		std::replace(searchName.begin(), searchName.end(), '.', '_');
	}
	std::filesystem::path outFileName = baseName + "." + searchName + ".Spe2Pep.txt";
	outPath = outPath / outFileName;
	sOutputFile = outPath.string();
}

bool MvhScanVector::ReadFT2File()
{
    MVH_PROFILE_SCOPE("mvh/function/MvhScanVector::ReadFT2File");
	bool bReVal, flag_1stScan = true; // flag_1stScan true indicates pMS2Scan is empty
	string sline;
	istringstream input;
	MS2Scan *pMS2Scan = NULL;
	ifstream ft2_stream(sFT2Filename.c_str());
	int tmp_charge;
	double tmp_mz, tmp_intensity;

	bool precursorChargeRead = false;

	bReVal = ft2_stream.is_open();
	if (bReVal)
	{
		while (!ft2_stream.eof())
		{
			sline.clear();
			//	    string tmp_sline = sline;
			getline(ft2_stream, sline);
			//	    cout << sline <<"wyf"<<endl;
			if (sline == "")
				continue;
			if ((sline.at(0) >= '0') && (sline.at(0) <= '9'))
			{
				TokenVector words(sline, " \t\n\r");
				if (words.size() < 6)
				// The judgement of MS scan resolution here will be overwritten,
				// we still keep the related codes here for future possible usuage

				{
					pMS2Scan->isMS2HighRes = false;
					pMS2Scan->viCharge.push_back(0);
				}
				else
				{
					pMS2Scan->isMS2HighRes = true;
					input.clear();
					input.str(words[5]);
					input >> tmp_charge;
					input.clear();
					pMS2Scan->viCharge.push_back(tmp_charge);
				}
				input.clear();
				input.str(words[0]);
				input >> tmp_mz;
				input.clear();
				if (tmp_mz > ProNovoConfig::maxObservedMz)
				{
					ProNovoConfig::maxObservedMz = tmp_mz;
				}
				if (tmp_mz < ProNovoConfig::minObservedMz)
				{
					ProNovoConfig::minObservedMz = tmp_mz;
				}
				pMS2Scan->vdMZ.push_back(tmp_mz);
				input.str(words[1]);
				input >> tmp_intensity;
				input.clear();
				pMS2Scan->vdIntensity.push_back(tmp_intensity);
			}
			else if (sline.at(0) == 'S')
			{
				if (flag_1stScan)
					flag_1stScan = false;
				else if (pMS2Scan->vdIntensity.empty())
					delete pMS2Scan;
				else
				{
					if (ProNovoConfig::getMassAccuracyFragmentIon() < 0.1)
						pMS2Scan->isMS2HighRes = true;
					else
						pMS2Scan->isMS2HighRes = false;
					saveFT2Scan(pMS2Scan);
				}
				pMS2Scan = new MS2Scan;
				pMS2Scan->sFT2Filename = sFT2Filename;
				TokenVector words(sline, " \r\t\n");
				input.clear();
				input.str(words[1]);
				input >> pMS2Scan->iScanId;
				input.clear();
				input.str(words[2]);
				input >> pMS2Scan->dParentMZ;
				input.clear();
				pMS2Scan->isMS1HighRes = isMS1HighRes(words[2]);
				// in case no Z Line
				pMS2Scan->iParentChargeState = 0;
				pMS2Scan->dParentNeutralMass = 0;
				precursorChargeRead = false;
			}
			else if (sline.at(0) == 'Z')
			{
				// Use only the first reported charge; S owns the precursor m/z.
				if (precursorChargeRead) continue;
				precursorChargeRead = true;
				TokenVector words(sline, " \r\t\n");
				input.clear();
				input.str(words.size() > 1 ? words[1] : "");
				if (!(input >> pMS2Scan->iParentChargeState) || pMS2Scan->iParentChargeState <= 0)
					throw std::runtime_error("Invalid instrument precursor charge in scan " + std::to_string(pMS2Scan->iScanId));
				input.clear();
			}
			else if (sline.at(0) == 'I')
			{
				TokenVector words(sline, " \r\t\n");
				if (words[1] == "ScanType")
					pMS2Scan->setScanType(words[2]);
				// retention time
				if (words[1] == "RetentionTime" || words[1] == "RTime")
				{
					pMS2Scan->setRTime(words.at(2));
				}
			}
			else if (sline.at(0) == 'D')
			{
				TokenVector words(sline, " \r\t\n");
				if (words[1] == "ParentScanNumber")
					pMS2Scan->iParentScanID = stoi(words[2]);
			}
		}
		ft2_stream.clear();
		ft2_stream.close();

		// recognition high or low MS2
		//	cout<<ProNovoConfig::getMassAccuracyFragmentIon()<<endl;
		if (ProNovoConfig::getMassAccuracyFragmentIon() < 0.1)
			pMS2Scan->isMS2HighRes = true;
		else
			pMS2Scan->isMS2HighRes = false;
		if (!flag_1stScan) // To avoid empty file
			saveFT2Scan(pMS2Scan);
	}
	return bReVal;
}

bool MvhScanVector::ReadMzmlFile()
{
    MVH_PROFILE_SCOPE("mvh/function/MvhScanVector::ReadMzmlFile");
	bool bReVal = false;
	MS2Scan *pMS2Scan;
	vector<Spectrum> *_vSpectra = new vector<Spectrum>();
	SiprosReader::MzmlReader(sFT2Filename, _vSpectra);
	int scannumber = _vSpectra->size();
	int peaknumber = 0;
	double tmp_mz = 0;
	for (int i = 0; i < scannumber; ++i)
	{
		// cout << _vSpectra->at(i).getScanNumber() << endl;
		pMS2Scan = new MS2Scan;
		pMS2Scan->sFT2Filename = sFT2Filename;
		pMS2Scan->iScanId = _vSpectra->at(i).getScanNumber();
		pMS2Scan->dParentMZ = _vSpectra->at(i).getMZ(0);
		// in case no Z Line
		pMS2Scan->iParentChargeState = 0;
		pMS2Scan->dParentNeutralMass = 0;

		if (_vSpectra->at(i).sizeZ() > 0)
		{
			pMS2Scan->iParentChargeState = _vSpectra->at(i).atZ(0).z;
		}

		pMS2Scan->setRTime(std::to_string((long double)_vSpectra->at(i).getRTime()));

		peaknumber = _vSpectra->at(i).getPeaks()->size();
		for (int j = 0; j < peaknumber; ++j)
		{
			tmp_mz = _vSpectra->at(i).getPeaks()->at(j).mz;
			if (tmp_mz > ProNovoConfig::maxObservedMz)
			{
				ProNovoConfig::maxObservedMz = tmp_mz;
			}
			if (tmp_mz < ProNovoConfig::minObservedMz)
			{
				ProNovoConfig::minObservedMz = tmp_mz;
			}
			pMS2Scan->vdMZ.push_back(_vSpectra->at(i).getPeaks()->at(j).mz);
			pMS2Scan->vdIntensity.push_back(_vSpectra->at(i).getPeaks()->at(j).intensity);
			pMS2Scan->viCharge.push_back(0);
		}

		if (ProNovoConfig::getMassAccuracyFragmentIon() < 0.1)
			pMS2Scan->isMS2HighRes = true;
		else
			pMS2Scan->isMS2HighRes = false;

		if (pMS2Scan->vdIntensity.empty())
		{
			delete pMS2Scan;
		}
		else
		{
			saveMzmlScan(pMS2Scan);
		}
	}
	_vSpectra->clear();
	delete _vSpectra;
	bReVal = true;
	return bReVal;
}

bool MvhScanVector::loadFT2File()
{
    MVH_PROFILE_SCOPE("mvh/function/MvhScanVector::loadFT2File");
	bool bReVal; // false when the file fails to be opened.
	double parentNeutralMass;
	bReVal = ReadFT2File();
	if (bReVal)
	{
		vAllPrecursorMassChargeMS2ScanPtrTuples.reserve(vpAllMS2Scans.size());
		vpPrecursorMasses.reserve(vpAllMS2Scans.size());
		for (size_t i = 0; i < vpAllMS2Scans.size(); i++)
		{
			MS2Scan *scan = vpAllMS2Scans[i];
			parentNeutralMass = scan->dParentMZ * scan->iParentChargeState -
				scan->iParentChargeState * ProNovoConfig::getProtonMass();
			vAllPrecursorMassChargeMS2ScanPtrTuples.push_back(
				{parentNeutralMass, scan->iParentChargeState, scan});
		}
		std::sort(vAllPrecursorMassChargeMS2ScanPtrTuples.begin(), vAllPrecursorMassChargeMS2ScanPtrTuples.end(),
				  [](const std::tuple<double, int, MS2Scan *> &a, const std::tuple<double, int, MS2Scan *> &b)
				  {
					  return get<0>(a) < get<0>(b);
				  });
		for (size_t i = 0; i < vAllPrecursorMassChargeMS2ScanPtrTuples.size(); i++)
		{
			vpPrecursorMasses.push_back(get<0>(vAllPrecursorMassChargeMS2ScanPtrTuples[i]));
		}
	}
#ifdef Ticktock
	TOCK1ST(loadFT2file);
#endif
	return bReVal;
}

bool MvhScanVector::loadMassData()
{
    MVH_PROFILE_SCOPE("mvh/function/MvhScanVector::loadMassData");
	CLOCKSTART;
	// read all MS2 scans from the file and populate vpAllMS2Scans
	// sort all MS2 scans in vpAllProteins by ascending order of their precursor masses
	// save their precursor masses in vpPrecursorMasses to quick look-up in assignPeptides2Scans()

	bool bReVal = false; // false when the file fails to be opened.
	vector<MS2Scan *>::iterator it;

	// check the suffix of the MS2 data file
	string filename_str = this->sFT2Filename;
	transform(filename_str.begin(), filename_str.end(), filename_str.begin(), (int (*)(int))tolower);
	string fileNameSuffix = filename_str.substr(filename_str.rfind('.') + 1);
	ProNovoConfig::getSetFileNameSuffix() = fileNameSuffix;
	if (fileNameSuffix == "mzml")
	{
		bReVal = ReadMzmlFile();
		if (bReVal)
		{
			sort(vpAllMS2Scans.begin(), vpAllMS2Scans.end(), myless);
			vAllPrecursorMassChargeMS2ScanPtrTuples.reserve(vpAllMS2Scans.size());
			vpPrecursorMasses.reserve(vpAllMS2Scans.size());
			for (it = vpAllMS2Scans.begin(); it < vpAllMS2Scans.end(); it++)
			{
				vpPrecursorMasses.push_back((*it)->dParentNeutralMass);
				vAllPrecursorMassChargeMS2ScanPtrTuples.push_back({(*it)->dParentNeutralMass,
																   (*it)->iParentChargeState,
																   (*it)});
			}
		}
	}
	else if (fileNameSuffix == "ft2")
	{
		bReVal = loadFT2File();
	}
	else
	{
		cout << "MS2 format not support!" << endl;
	}

	double mass = 0;
	int charge = 0;
	MS2Scan *pMS2Scan;
	for (const auto &tuple : vAllPrecursorMassChargeMS2ScanPtrTuples)
	{
		tie(mass, charge, pMS2Scan) = tuple;
		// set max Parent Neutral Mass in isolation window of each MS2Scan for MVH score function
		if (pMS2Scan->dParentNeutralMass < mass)
			pMS2Scan->dParentNeutralMass = mass;
		// set max parent mass in isolation window of each MS2Scan for Xcorr score function
		mass = mass + charge * ProNovoConfig::getProtonMass();
		if (pMS2Scan->dParentMass < mass)
			pMS2Scan->dParentMass = mass;
		// set max precursor mass for Xcorr score function
		if (ProNovoConfig::dMaxMS2ScanMass < mass)
			ProNovoConfig::dMaxMS2ScanMass = mass;
		// set max precursor charge for Xcorr score function
		if (ProNovoConfig::iMaxPercusorCharge < charge)
			ProNovoConfig::iMaxPercusorCharge = charge;
	}
	cout << "\nload mass data done.\n"
		 << endl;
	CLOCKSTOP;
	return bReVal;
}

bool MvhScanVector::isMS1HighRes(const std::string &target)
{
	bool bReVal = true;
	if (target.at(target.size() - 2) == '.')
		bReVal = false;
	return bReVal;
}

bool MvhScanVector::ChargeDetermination(const std::vector<double> &vdAllmz, double pmz)
// return true, if the charge state is decided to be one

{
	bool bReVal = false;
	int iPosi;
	vector<double> vdtempMZ = vdAllmz;
	iPosi = max(0, (int)(((int)vdtempMZ.size()) * 0.05 - 1));
	nth_element(vdtempMZ.begin(), vdtempMZ.begin() + iPosi, vdtempMZ.end(), mygreater);
	if (vdtempMZ.at(iPosi) < pmz)
		bReVal = true;
	return bReVal;
}

bool MvhScanVector::mygreater(double i, double j)
{
	return (i > j);
}

bool MvhScanVector::myless(MS2Scan *pMS2Scan1, MS2Scan *pMS2Scan2)
{
	return (pMS2Scan1->dParentNeutralMass < pMS2Scan2->dParentNeutralMass);
}

void MvhScanVector::saveFT2Scan(MS2Scan *pMS2Scan)
{
	if (pMS2Scan->iParentChargeState <= 0 || !(pMS2Scan->dParentMZ > 0))
		throw std::runtime_error("Missing or invalid instrument precursor in scan " + std::to_string(pMS2Scan->iScanId));
	pMS2Scan->dParentNeutralMass = pMS2Scan->dParentMZ * pMS2Scan->iParentChargeState -
		pMS2Scan->iParentChargeState * ProNovoConfig::getProtonMass();
	vpAllMS2Scans.push_back(pMS2Scan);
}

void MvhScanVector::saveMzmlScan(MS2Scan *pMS2Scan)
{ // parentChargeState > 0, save, otherwise, try 1, or 2 and 3
	int j;
	bool bchargeOne;
	MS2Scan *pMS2newScan;
	if (pMS2Scan->iParentChargeState == 0)
	{
		bchargeOne = ChargeDetermination(pMS2Scan->vdMZ, pMS2Scan->dParentMZ);
		if (bchargeOne)
		{
			j = 1;
			pMS2newScan = new MS2Scan;
			*pMS2newScan = *pMS2Scan;
			pMS2newScan->iParentChargeState = j;
			pMS2newScan->dParentNeutralMass = (pMS2newScan->dParentMZ) * (pMS2newScan->iParentChargeState) - pMS2newScan->iParentChargeState * ProNovoConfig::getProtonMass();
			vpAllMS2Scans.push_back(pMS2newScan);
		}
		else
		// if the charge state is not +1, it could be greater than +3,
		// but current we just try +2 and +3.
		{
			for (j = 2; j <= 3; j++)
			{
				pMS2newScan = new MS2Scan;
				*pMS2newScan = *pMS2Scan;
				pMS2newScan->iParentChargeState = j;
				pMS2newScan->dParentNeutralMass = (pMS2newScan->dParentMZ) * (pMS2newScan->iParentChargeState) - pMS2newScan->iParentChargeState * ProNovoConfig::getProtonMass();
				vpAllMS2Scans.push_back(pMS2newScan);
			}
		}
	}
	else
	{
		pMS2newScan = new MS2Scan;
		*pMS2newScan = *pMS2Scan;
		pMS2newScan->dParentNeutralMass = (pMS2newScan->dParentMZ) * (pMS2newScan->iParentChargeState) - (pMS2newScan->iParentChargeState) * ProNovoConfig::getProtonMass();
		pMS2newScan->dParentMass = (pMS2newScan->dParentMZ) * (pMS2newScan->iParentChargeState);
		vpAllMS2Scans.push_back(pMS2newScan);
	}
	delete pMS2Scan;
}

void MvhScanVector::preProcessAllMs2Mvh()
{
    MVH_PROFILE_SCOPE("mvh/function/MvhScanVector::preProcessAllMs2Mvh");
    mvh_cuda::preProcessAllMs2Mvh(vpAllMS2Scans);
}
