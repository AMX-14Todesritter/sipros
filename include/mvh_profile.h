#ifndef MVH_PROFILE_H
#define MVH_PROFILE_H

#ifdef SIPROS_MVH_PROFILE

#include <string>

class MS2Scan;
class PeakList;

namespace MvhProfile
{

struct InspectNearResult
{
	bool hit;
	int matched_peak_index;
	double matched_peak_mz;
	double mass_error;
	char mvh_class;
	int num_peaks_in_window;
};

bool shouldProfileScan(const MS2Scan *scan);
void writeObservedPeaks(const MS2Scan *scan);
void beginCandidate(int scan_id, int candidate_index);
InspectNearResult inspectNear(const PeakList *peakList, double query_mz, double tolerance);
void recordFragmentQuery(const MS2Scan *scan, const std::string &peptide_sequence,
						 int precursor_charge, int fragment_index, double theoretical_mz,
						 double tolerance, bool in_spectrum_range, char production_class);
void writeScanSummary(const MS2Scan *scan, int candidate_count);

} // namespace MvhProfile

#endif // SIPROS_MVH_PROFILE

#endif // MVH_PROFILE_H
