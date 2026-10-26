    if (peakAuditBuffer) {
        std::vector<PeakAuditState> auditHost;peakAuditBuffer->read(auditHost);
        const auto& audit=auditHost[0];unsigned long long sums[7]{};
        for(const auto& shard:audit.shards)for(int k=0;k<7;++k)sums[k]+=shard[k];
        static unsigned long long auditBatch=0;++auditBatch;
        const std::filesystem::path dir(std::getenv("SIPROS_PEAK_AUDIT_DIR"));
        std::filesystem::create_directories(dir);
        const auto summaryPath=dir/"peak_audit_batches.tsv";
        bool first=!std::filesystem::exists(summaryPath);
        std::ofstream summary(summaryPath,std::ios::app);summary.exceptions(std::ios::failbit|std::ios::badbit);
        if(first)summary<<"batch\tqueries\toracle_hits\toracle_misses\tpeak_identity_mismatches\tclass_mismatches\tactual_hits\tactual_misses\tmismatch_examples_total\n";
        summary<<auditBatch;for(auto n:sums)summary<<'\t'<<n;summary<<'\t'<<audit.exampleCount<<'\n';
        const auto examplePath=dir/"peak_audit_examples.tsv";first=!std::filesystem::exists(examplePath);
        std::ofstream examples(examplePath,std::ios::app);examples.exceptions(std::ios::failbit|std::ios::badbit);examples<<std::setprecision(17);
        if(first)examples<<"batch\tscan_id\tscan_index\tpeptide\tprecursor_charge\tquery_index\ttheoretical_mz\ttolerance\toracle_retained_peak_id\tactual_retained_peak_id\toracle_class\tactual_class\toracle_mz\tactual_mz\n";
        static std::set<int> exported;
        const auto spectraPath=dir/"peak_audit_spectra.tsv";first=!std::filesystem::exists(spectraPath);
        std::ofstream observed(spectraPath,std::ios::app);observed.exceptions(std::ios::failbit|std::ios::badbit);observed<<std::setprecision(17);
        if(first)observed<<"scan_id\tretained_peak_id\tmz\tclass\n";
        for(unsigned long long j=0;j<std::min(audit.exampleCount,128ULL);++j){
            const auto& ex=audit.examples[j];Candidate candidate{};
            check(cudaMemcpy(&candidate,candidates.p+ex.candidate,sizeof(candidate),cudaMemcpyDeviceToHost));
            const auto sid=batch.spectra->owners.at(ex.scanIndex)->iScanId;
            examples<<auditBatch<<'\t'<<sid<<'\t'<<ex.scanIndex<<'\t'<<batch.peptideObjects.at(candidate.peptideId)->sPeptide<<'\t'<<candidate.charge
                <<'\t'<<ex.queryIndex<<'\t'<<ex.mz<<'\t'<<ex.tolerance<<'\t'<<ex.expectedPeak<<'\t'<<ex.actualPeak
                <<'\t'<<ex.expectedClass<<'\t'<<ex.actualClass<<'\t'<<ex.expectedMz<<'\t'<<ex.actualMz<<'\n';
            if(exported.insert(ex.scanIndex).second){const auto& scan=batch.scans.at(ex.scanIndex);
                for(int k=0;k<scan.peaks;++k)observed<<sid<<'\t'<<k<<'\t'<<batch.spectra->peaks.at(scan.peakOffset+k)<<'\t'<<batch.spectra->classes.at(scan.peakOffset+k)<<'\n';}
        }
    }
