#include "mvh_profile.h"

#ifdef SIPROS_MVH_PROFILE

#include "ms2scan.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

struct ScanCounters
{
	long long total_fragment_queries = 0;
	long long in_range_queries = 0;
	long long matched_queries = 0;
};

struct ThreadState
{
	int current_scan_id = -1;
	int current_candidate_index = -1;
	std::map<int, ScanCounters> scan_counters;
	std::ofstream scan_summary;
	std::ofstream observed_peaks;
	std::ofstream fragment_queries;
	bool scan_summary_header = false;
	bool observed_peaks_header = false;
	bool fragment_queries_header = false;
};

thread_local ThreadState g_state;

int threadId()
{
#ifdef _OPENMP
	return omp_get_thread_num();
#else
	return 0;
#endif
}

const std::set<int> &selectedScanIds()
{
	static const std::set<int> ids = []() {
		std::set<int> parsed_ids;
		const char *env = std::getenv("SIPROS_MVH_PROFILE_SCAN_IDS");
		if (env == NULL || env[0] == '\0')
		{
			return parsed_ids;
		}

		std::stringstream ss(env);
		std::string token;
		while (std::getline(ss, token, ','))
		{
			std::stringstream item(token);
			int scan_id = 0;
			if (item >> scan_id)
			{
				parsed_ids.insert(scan_id);
			}
		}
		return parsed_ids;
	}();
	return ids;
}

std::string csvEscape(const std::string &value)
{
	bool needs_quotes = false;
	for (char c : value)
	{
		if (c == ',' || c == '"' || c == '\n' || c == '\r')
		{
			needs_quotes = true;
			break;
		}
	}
	if (!needs_quotes)
	{
		return value;
	}

	std::string escaped = "\"";
	for (char c : value)
	{
		if (c == '"')
		{
			escaped += "\"\"";
		}
		else
		{
			escaped += c;
		}
	}
	escaped += "\"";
	return escaped;
}

std::ofstream &openThreadFile(std::ofstream &stream, const std::string &base_name)
{
	if (!stream.is_open())
	{
		std::ostringstream filename;
		filename << base_name << ".thread_" << threadId() << ".csv";
		stream.open(filename.str().c_str(), std::ios::out);
		stream << std::setprecision(12);
	}
	return stream;
}

bool productionHit(char production_class)
{
	return production_class != PeakList::iNULL && production_class > 0;
}

void failValidation(const MS2Scan *scan, int fragment_index, double theoretical_mz,
					double tolerance, char production_class,
					const MvhProfile::InspectNearResult &inspection,
					const std::string &message)
{
	std::cerr << "SIPROS_MVH_PROFILE validation failed: " << message << std::endl
			  << "  scan_id=" << scan->iScanId << std::endl
			  << "  candidate_index=" << g_state.current_candidate_index << std::endl
			  << "  fragment_index=" << fragment_index << std::endl
			  << "  theoretical_mz=" << theoretical_mz << std::endl
			  << "  tolerance=" << tolerance << std::endl
			  << "  production_class=" << static_cast<int>(production_class) << std::endl
			  << "  profiler_class=" << static_cast<int>(inspection.mvh_class) << std::endl;
	std::exit(1);
}

} // namespace

namespace MvhProfile
{

bool shouldProfileScan(const MS2Scan *scan)
{
	if (scan == NULL)
	{
		return false;
	}
	const std::set<int> &ids = selectedScanIds();
	return !ids.empty() && ids.find(scan->iScanId) != ids.end();
}

void writeObservedPeaks(const MS2Scan *scan)
{
	if (!shouldProfileScan(scan) || scan->pPeakList == NULL)
	{
		return;
	}

	std::ofstream &out = openThreadFile(g_state.observed_peaks, "observed_peaks");
	if (!g_state.observed_peaks_header)
	{
		out << "scan_id,peak_index,mz,intensity,mvh_retained,mvh_class\n";
		g_state.observed_peaks_header = true;
	}

	std::map<double, char> retained_by_mz;
	for (int i = 0; i < scan->pPeakList->iPeakSize; ++i)
	{
		retained_by_mz[scan->pPeakList->pPeaks.at(i)] = scan->pPeakList->pClasses.at(i);
	}

	for (int i = 0; i < (int)scan->vdMZ.size(); ++i)
	{
		bool retained = false;
		char mvh_class = 0;
		std::map<double, char>::iterator found = retained_by_mz.find(scan->vdMZ.at(i));
		if (found != retained_by_mz.end())
		{
			retained = true;
			mvh_class = found->second;
			retained_by_mz.erase(found);
		}

		out << scan->iScanId << ','
			<< i << ','
			<< scan->vdMZ.at(i) << ','
			<< scan->vdIntensity.at(i) << ','
			<< (retained ? 1 : 0) << ','
			<< static_cast<int>(mvh_class) << '\n';
	}
}

void beginCandidate(int scan_id, int candidate_index)
{
	g_state.current_scan_id = scan_id;
	g_state.current_candidate_index = candidate_index;
}

InspectNearResult inspectNear(const PeakList *peakList, double query_mz, double tolerance)
{
	InspectNearResult result;
	result.hit = false;
	result.matched_peak_index = -1;
	result.matched_peak_mz = 0.0;
	result.mass_error = 0.0;
	result.mvh_class = PeakList::iNULL;
	result.num_peaks_in_window = 0;

	if (peakList == NULL || peakList->iPeakSize == 0)
	{
		return result;
	}

	double dMin = 1000000;
	double dDiff = 0;
	int iMzU = (int)(query_mz + tolerance);
	int iMzL = (int)(query_mz - tolerance);
	if (iMzU < peakList->iLowestMass)
	{
		return result;
	}
	if (iMzL > peakList->iHighestMass)
	{
		return result;
	}

	int iStart = 0;
	int iEnd = peakList->iMassHubPairSizeMinusOne;
	if (iMzL >= peakList->iLowestMass)
	{
		iStart = iMzL - peakList->iLowestMass;
	}
	if (iMzU <= peakList->iHighestMass)
	{
		iEnd = iMzU - peakList->iLowestMass;
	}

	int best_index = -1;
	for (; iStart <= iEnd; ++iStart)
	{
		if (peakList->pMassHub.at(iStart * 2) != -1)
		{
			for (int i = peakList->pMassHub.at(iStart * 2), j = peakList->pMassHub.at(iStart * 2 + 1); i < j; ++i)
			{
				dDiff = fabs(query_mz - peakList->pPeaks.at(i));
				if (dDiff < tolerance)
				{
					++result.num_peaks_in_window;
				}
				if (dDiff < dMin)
				{
					dMin = dDiff;
					best_index = i;
					result.mvh_class = peakList->pClasses.at(i);
				}
			}
		}
	}

	if (dMin < tolerance)
	{
		result.hit = true;
		result.matched_peak_index = best_index;
		result.matched_peak_mz = peakList->pPeaks.at(best_index);
		result.mass_error = result.matched_peak_mz - query_mz;
	}
	else
	{
		result.mvh_class = PeakList::iNULL;
	}

	return result;
}

void recordFragmentQuery(const MS2Scan *scan, const std::string &peptide_sequence,
						 int precursor_charge, int fragment_index, double theoretical_mz,
						 double tolerance, bool in_spectrum_range, char production_class)
{
	if (!shouldProfileScan(scan))
	{
		return;
	}

	ScanCounters &counters = g_state.scan_counters[scan->iScanId];
	++counters.total_fragment_queries;

	InspectNearResult inspection;
	inspection.hit = false;
	inspection.matched_peak_index = -1;
	inspection.matched_peak_mz = 0.0;
	inspection.mass_error = 0.0;
	inspection.mvh_class = PeakList::iNULL;
	inspection.num_peaks_in_window = 0;

	if (in_spectrum_range)
	{
		++counters.in_range_queries;
		inspection = inspectNear(scan->pPeakList, theoretical_mz, tolerance);
		bool prod_hit = productionHit(production_class);

		if (inspection.hit != prod_hit)
		{
			failValidation(scan, fragment_index, theoretical_mz, tolerance, production_class,
						   inspection, "profiler hit/miss disagrees with PeakList::findNear()");
		}
		if (prod_hit && inspection.mvh_class != production_class)
		{
			failValidation(scan, fragment_index, theoretical_mz, tolerance, production_class,
						   inspection, "profiler class disagrees with PeakList::findNear()");
		}
		if (inspection.hit && fabs(theoretical_mz - inspection.matched_peak_mz) >= tolerance)
		{
			failValidation(scan, fragment_index, theoretical_mz, tolerance, production_class,
						   inspection, "matched peak is outside strict tolerance");
		}
		if (inspection.hit && inspection.matched_peak_index < 0)
		{
			failValidation(scan, fragment_index, theoretical_mz, tolerance, production_class,
						   inspection, "hit has no matched peak index");
		}
		if (!inspection.hit && inspection.matched_peak_index != -1)
		{
			failValidation(scan, fragment_index, theoretical_mz, tolerance, production_class,
						   inspection, "miss has a matched peak index");
		}
		if (inspection.hit && inspection.num_peaks_in_window <= 0)
		{
			failValidation(scan, fragment_index, theoretical_mz, tolerance, production_class,
						   inspection, "hit has no peaks inside tolerance window");
		}

		if (prod_hit)
		{
			++counters.matched_queries;
		}
	}

	std::ofstream &out = openThreadFile(g_state.fragment_queries, "fragment_queries");
	if (!g_state.fragment_queries_header)
	{
		out << "scan_id,candidate_index,peptide_sequence,precursor_charge,fragment_index,theoretical_mz,tolerance,"
			<< "in_spectrum_range,hit,matched_peak_index,matched_peak_mz,mass_error,mvh_class,num_peaks_in_window\n";
		g_state.fragment_queries_header = true;
	}

	out << scan->iScanId << ','
		<< g_state.current_candidate_index << ','
		<< csvEscape(peptide_sequence) << ','
		<< precursor_charge << ','
		<< fragment_index << ','
		<< theoretical_mz << ','
		<< tolerance << ','
		<< (in_spectrum_range ? 1 : 0) << ','
		<< (inspection.hit ? 1 : 0) << ','
		<< inspection.matched_peak_index << ',';
	if (inspection.hit)
	{
		out << inspection.matched_peak_mz << ','
			<< inspection.mass_error << ','
			<< static_cast<int>(inspection.mvh_class) << ',';
	}
	else
	{
		out << ",,0,";
	}
	out << inspection.num_peaks_in_window << '\n';
}

void writeScanSummary(const MS2Scan *scan, int candidate_count)
{
	if (!shouldProfileScan(scan))
	{
		return;
	}

	ScanCounters counters = g_state.scan_counters[scan->iScanId];
	long long missed = counters.in_range_queries - counters.matched_queries;
	if (missed < 0 || counters.matched_queries + missed != counters.in_range_queries)
	{
		std::cerr << "SIPROS_MVH_PROFILE validation failed: inconsistent scan counters" << std::endl
				  << "  scan_id=" << scan->iScanId << std::endl
				  << "  matched_queries=" << counters.matched_queries << std::endl
				  << "  missed_queries=" << missed << std::endl
				  << "  in_range_queries=" << counters.in_range_queries << std::endl;
		std::exit(1);
	}

	int processed_peak_count = scan->pPeakList == NULL ? 0 : scan->pPeakList->size();
	double hit_rate = counters.in_range_queries > 0
						  ? (double)counters.matched_queries / (double)counters.in_range_queries
						  : 0.0;

	std::ofstream &out = openThreadFile(g_state.scan_summary, "scan_summary");
	if (!g_state.scan_summary_header)
	{
		out << "scan_id,ft2_filename,retention_time,precursor_mz,precursor_neutral_mass,raw_peak_count,"
			<< "processed_peak_count,candidate_count,total_fragment_queries,in_range_queries,matched_queries,"
			<< "missed_queries,hit_rate\n";
		g_state.scan_summary_header = true;
	}

	out << scan->iScanId << ','
		<< csvEscape(scan->sFT2Filename) << ','
		<< csvEscape(scan->sRTime) << ','
		<< scan->dParentMZ << ','
		<< scan->dParentNeutralMass << ','
		<< scan->vdMZ.size() << ','
		<< processed_peak_count << ','
		<< candidate_count << ','
		<< counters.total_fragment_queries << ','
		<< counters.in_range_queries << ','
		<< counters.matched_queries << ','
		<< missed << ','
		<< hit_rate << '\n';

	g_state.scan_counters.erase(scan->iScanId);
}

} // namespace MvhProfile

#endif // SIPROS_MVH_PROFILE
