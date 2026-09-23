#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include "ms2scan.h"
#include "tokenvector.h"
#include "proNovoConfig.h"
#include "proteindatabase.h"
#include "MVH.h"

#define PEPTIDE_ARRAY_SIZE 2000000

// MVH-only search hierarchy. Assignment is batched on CUDA; original CPU
// method names remain available for tracing the search stages.
class MvhScanVector {
public:
    MvhScanVector(const string &sFT2FilenameInput, const string &sOutputDirectory,
                  const string &sConfigFilename, bool bScreenOutput);
    ~MvhScanVector();

    bool loadMassData();
    void preProcessAllMs2Mvh();
    void searchDatabaseMvh();

    // Original member name; the application reads retained results after search.
    vector<MS2Scan *> vpAllMS2Scans;

private:
    vector<double> vpPrecursorMasses;
    vector<tuple<double, int, MS2Scan *>> vAllPrecursorMassChargeMS2ScanPtrTuples;
    string sFT2Filename;
    string sOutputFile;
    string sConfigFile;
    map<char, double> mapResidueMass;
    bool bScreenOutput;
    // The unchanged constructor initializes this original member.
    int iOpenMPTaskNum;

    bool ReadFT2File();
    bool ReadMzmlFile();
    bool loadFT2File();
    void setOutputFile(const string &sFT2FilenameInput, const string &sOutputDirectory);
    void saveFT2Scan(MS2Scan *pMS2Scan);
    void saveMzmlScan(MS2Scan *pMS2Scan);
    bool isMS1HighRes(const string &target);
    bool ChargeDetermination(const vector<double> &vdAllmz, double pmz);
    static bool mygreater(double i, double j);
    static bool myless(MS2Scan *pMS2Scan1, MS2Scan *pMS2Scan2);

    void assignPeptides2Scans(const vector<Peptide *> &peptides);
    pair<int, int> GetRangeFromMass(double lb, double ub);
    void GetAllRangeFromMass(double dPeptideMass, vector<pair<int, int>> &vpPeptideMassRanges);
    void processPeptideArrayMvh(vector<Peptide *> &vpPeptideArray);

    vector<vector<double> *> _ppdAAforward;
    vector<vector<double> *> _ppdAAreverse;
    vector<vector<double> *> psequenceIonMasses;
    vector<vector<char> *> pSeqs;
    int num_max_threads;
    void preMvh();
    void postMvh();
};
